param(
    [string]$Location = "eastus",
    [string]$FrontendOrigin = "https://footee-highlights.vercel.app"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI is not installed. Run: winget install --exact --id Microsoft.AzureCLI"
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop is not installed. Install or start Docker Desktop before deploying."
}

$account = az account show --output json 2>$null | ConvertFrom-Json
if (-not $account) {
    az login | Out-Null
    $account = az account show --output json | ConvertFrom-Json
}

if ($account.name -notmatch "Student") {
    Write-Warning "Active subscription is '$($account.name)'. Confirm this is the Azure for Students subscription before continuing."
}

$subscriptionSuffix = ($account.id -replace "-", "").Substring(0, 8).ToLowerInvariant()
$resourceGroup = "footee-vision-rg"
$environment = "footee-vision-env"
$appName = "footee-vision-api"
$registryName = "footeevision$subscriptionSuffix"
$imageName = "footee-vision-backend"
$commitTag = (git rev-parse --short HEAD 2>$null)
if (-not $commitTag) { $commitTag = "build" }
$imageTag = "$commitTag-$(Get-Date -Format 'yyyyMMddHHmmss')"

Write-Host "Using subscription: $($account.name)"
Write-Host "Registering Azure providers..."
az provider register --namespace Microsoft.App --wait | Out-Null
az provider register --namespace Microsoft.OperationalInsights --wait | Out-Null
az provider register --namespace Microsoft.ContainerRegistry --wait | Out-Null

az extension add --name containerapp --upgrade --yes | Out-Null
az group create --name $resourceGroup --location $Location | Out-Null

$registryExists = az acr list `
    --resource-group $resourceGroup `
    --query "[?name=='$registryName'].name | [0]" `
    --output tsv
if (-not $registryExists) {
    az acr create `
        --name $registryName `
        --resource-group $resourceGroup `
        --sku Basic `
        --admin-enabled true | Out-Null
}

$loginServer = az acr show --name $registryName --query loginServer --output tsv

# Azure for Students can block ACR Tasks/cloud builds. Build with the local
# Linux Docker engine and push to the registry instead.
Write-Host "Signing Docker in to Azure Container Registry..."
az acr login --name $registryName
if ($LASTEXITCODE -ne 0) { throw "Azure Container Registry login failed" }

Write-Host "Building the CPU image locally. This can take several minutes..."
docker build --tag "${loginServer}/${imageName}:${imageTag}" .
if ($LASTEXITCODE -ne 0) { throw "Docker image build failed" }

docker push "${loginServer}/${imageName}:${imageTag}"
if ($LASTEXITCODE -ne 0) { throw "Docker image push failed" }

$environmentExists = az containerapp env list `
    --resource-group $resourceGroup `
    --query "[?name=='$environment'].name | [0]" `
    --output tsv
if (-not $environmentExists) {
    az containerapp env create `
        --name $environment `
        --resource-group $resourceGroup `
        --location $Location | Out-Null
}

$environmentState = az containerapp env show `
    --name $environment `
    --resource-group $resourceGroup `
    --query properties.provisioningState `
    --output tsv
while ($environmentState -notin @("Succeeded", "Failed")) {
    Write-Host "Waiting for the Container Apps environment ($environmentState)..."
    Start-Sleep -Seconds 15
    $environmentState = az containerapp env show `
        --name $environment `
        --resource-group $resourceGroup `
        --query properties.provisioningState `
        --output tsv
}
if ($environmentState -eq "Failed") {
    throw "Azure Container Apps environment provisioning failed"
}

$registryUsername = az acr credential show --name $registryName --query username --output tsv
$registryPassword = az acr credential show --name $registryName --query "passwords[0].value" --output tsv
$appExists = az containerapp list `
    --resource-group $resourceGroup `
    --query "[?name=='$appName'].name | [0]" `
    --output tsv

$environmentVariables = @(
    "CORS_ORIGINS=$($FrontendOrigin.TrimEnd('/'))",
    "FOOTEE_STORAGE_DIR=/tmp/footee-vision",
    "LOW_MEMORY_MODE=true",
    "SCENE_DETECTION_METHOD=transnetv2",
    "TRANSNETV2_WINDOW_SIZE=50",
    "TRANSNETV2_CPU_THREADS=2",
    "YOLO_MODEL_PATH=yolo11n.pt",
    "TRACKING_MODEL_PATH=yolo11n.pt",
    "TRACKING_BATCH_SIZE=4",
    "MAX_UPLOAD_SIZE_MB=500",
    "UPLOAD_RETENTION_SECONDS=3600",
    "OMP_NUM_THREADS=2",
    "MKL_NUM_THREADS=2",
    "OPENBLAS_NUM_THREADS=2",
    "MALLOC_ARENA_MAX=2"
)

if (-not $appExists) {
    az containerapp create `
        --name $appName `
        --resource-group $resourceGroup `
        --environment $environment `
        --image "${loginServer}/${imageName}:${imageTag}" `
        --registry-server $loginServer `
        --registry-username $registryUsername `
        --registry-password $registryPassword `
        --ingress external `
        --target-port 8000 `
        --transport auto `
        --cpu 2.0 `
        --memory 4Gi `
        --min-replicas 0 `
        --max-replicas 1 `
        --scale-rule-http-concurrency 1 `
        --env-vars $environmentVariables | Out-Null
} else {
    az containerapp registry set `
        --name $appName `
        --resource-group $resourceGroup `
        --server $loginServer `
        --username $registryUsername `
        --password $registryPassword | Out-Null
    az containerapp update `
        --name $appName `
        --resource-group $resourceGroup `
        --image "${loginServer}/${imageName}:${imageTag}" `
        --cpu 2.0 `
        --memory 4Gi `
        --min-replicas 0 `
        --max-replicas 1 `
        --set-env-vars $environmentVariables | Out-Null
}

$fqdn = az containerapp show `
    --name $appName `
    --resource-group $resourceGroup `
    --query properties.configuration.ingress.fqdn `
    --output tsv

Write-Host ""
Write-Host "Backend URL: https://$fqdn"
Write-Host "Health URL:  https://$fqdn/health"
Write-Host "Set VITE_API_BASE_URL=https://$fqdn in Vercel, then redeploy the frontend."
Write-Host ""
Write-Host "To avoid accidental charges after testing, remove everything with:"
Write-Host "az group delete --name $resourceGroup --yes --no-wait"
