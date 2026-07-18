import type { Footedness } from "../types/analysis";

type FootedRecommendations = Record<Footedness, readonly string[]>;

const PLAYER_RECOMMENDATIONS: Record<string, FootedRecommendations> = {
  holding_six: {
    left: ["Aurelien Tchouameni", "Stefan Bajcetic", "Nemanja Matic"],
    right: ["Declan Rice", "Moises Caicedo", "Martin Zubimendi"],
    both: ["Amadou Onana", "Boubacar Kamara", "Andre-Frank Zambo Anguissa"],
  },
  deep_lying_playmaker: {
    left: ["Eduardo Camavinga", "Granit Xhaka", "Ismael Bennacer"],
    right: ["Rodri", "Vitinha", "Aleksandar Pavlovic"],
    both: ["Frenkie de Jong", "Joshua Kimmich", "Sergio Busquets"],
  },
  box_to_box: {
    left: ["Martin Odegaard", "Mikel Merino", "Fabian Ruiz"],
    right: ["Jude Bellingham", "Enzo Fernandez", "Dominik Szoboszlai"],
    both: ["Federico Valverde", "Ilkay Gundogan", "Luka Modric"],
  },
  attacking_playmaker: {
    left: ["Bernardo Silva", "Cole Palmer", "Lucas Paqueta"],
    right: ["Bruno Fernandes", "Florian Wirtz", "Xavi Simons"],
    both: ["Jamal Musiala", "Dani Olmo", "Pedri"],
  },
  target_man: {
    left: ["Erling Haaland", "Benjamin Sesko", "Romelu Lukaku"],
    right: ["Dominic Solanke", "Jean-Philippe Mateta", "Viktor Gyokeres"],
    both: ["Harry Kane", "Marcus Thuram"],
  },
  poacher: {
    left: ["Lois Openda", "Serhou Guirassy", "Santiago Gimenez"],
    right: ["Jamie Vardy", "Victor Osimhen", "Lautaro Martinez"],
    both: ["Cristiano Ronaldo", "Robert Lewandowski", "Diogo Jota"],
  },
  inside_forward: {
    left: ["Mohamed Salah", "Michael Olise", "Arjen Robben"],
    right: ["Kylian Mbappe", "Vinicius Junior"],
    both: ["Heung-min Son", "Khvicha Kvaratskhelia", "Ousmane Dembele"],
  },
  classic_winger: {
    left: ["Leroy Sane", "Estevao", "Bukayo Saka"],
    right: ["Bradley Barcola", "Yan Diomande", "Antonio Nusa"],
    both: ["Luis Diaz", "Kaoru Mitoma", "Eden Hazard"],
  },
  false_nine: {
    left: ["Paulo Dybala", "Mikel Oyarzabal", "Lionel Messi"],
    right: ["Julian Alvarez", "Joao Pedro", "Nicolas Jackson"],
    both: ["Roberto Firmino", "Ousmane Dembele"],
  },
};

function isFootedness(value: string | undefined): value is Footedness {
  return value === "left" || value === "right" || value === "both";
}

export function getSimilarPlayers(archetype: string, footedness: string | undefined) {
  if (!isFootedness(footedness)) return [];

  const recommendations = PLAYER_RECOMMENDATIONS[archetype]?.[footedness] ?? [];
  return [...new Set(recommendations)];
}
