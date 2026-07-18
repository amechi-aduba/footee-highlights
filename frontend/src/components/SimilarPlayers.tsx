import { getSimilarPlayers } from "../data/playerRecommendations";

interface SimilarPlayersProps {
  archetype: string;
  footedness: string | undefined;
}

export function SimilarPlayers({ archetype, footedness }: SimilarPlayersProps) {
  const players = getSimilarPlayers(archetype, footedness);

  if (players.length === 0) return null;

  return (
    <aside className="relative overflow-hidden rounded-2xl border border-primary/25 bg-primary/[0.06] p-5 lg:sticky lg:top-20">
      <span
        aria-hidden
        className="absolute -right-8 -top-8 h-24 w-24 rounded-full bg-primary/15 blur-2xl"
      />
      <p className="kicker">Film study</p>
      <h3 className="mt-2 text-lg font-extrabold tracking-tight">Players to watch</h3>
      <p className="mt-1.5 text-xs leading-relaxed text-mute">
        Watch film of these players to study how they bring this role to life.
      </p>
      <ol className="mt-4 space-y-2.5">
        {players.map((player, index) => (
          <li
            key={player}
            className="flex items-center gap-3 rounded-xl border border-line/80 bg-surface/80 px-3 py-2.5 shadow-card"
          >
            <span className="font-mono text-[10px] font-bold text-primary">
              {String(index + 1).padStart(2, "0")}
            </span>
            <span className="text-xs font-bold leading-tight text-ink">{player}</span>
          </li>
        ))}
      </ol>
    </aside>
  );
}
