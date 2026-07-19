export interface DemoSample {
  id: string;
  title: string;
  subtitle: string;
  duration: string;
  resultPath: string;
  posterPath: string;
}

export const DEMO_SAMPLES: readonly DemoSample[] = [
  {
    id: "alistair-johnston",
    title: "Alistair Johnston",
    subtitle: "2019 Wake Forest highlights",
    duration: "5:14",
    resultPath: "/samples/alistair-johnston/result.json",
    posterPath: "/samples/alistair-johnston/thumbnails/seg_002.jpg",
  },
  {
    id: "jack-harrison",
    title: "Jack Harrison",
    subtitle: "Number 11 highlights",
    duration: "6:29",
    resultPath: "/samples/jack-harrison/result.json",
    posterPath: "/samples/jack-harrison/thumbnails/seg_001.jpg",
  },
  {
    id: "ousseni-bouda",
    title: "Ousseni Bouda",
    subtitle: "Senior year high-school highlights",
    duration: "8:47",
    resultPath: "/samples/ousseni-bouda/result.json",
    posterPath: "/samples/ousseni-bouda/thumbnails/seg_001.jpg",
  },
];
