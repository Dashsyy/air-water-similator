import type { Snapshot } from './simulation';

export interface ChallengeContext {
  snapshot: Snapshot;
  modelName: string;
  imported: boolean;
  /** First settled drag coefficient recorded for this model in this session, if any. */
  firstCd: number | null;
}

export interface Challenge {
  id: string;
  title: string;
  hint: string;
  emoji: string;
  passed: (c: ChallengeContext) => boolean;
}

export const CHALLENGES: Challenge[] = [
  {
    id: 'first-run',
    title: 'First measurement',
    hint: 'Let any object sit in the wind until its numbers settle.',
    emoji: '🌬️',
    passed: () => true,
  },
  {
    id: 'beat-brick',
    title: 'Beat the box',
    hint: 'Settle a drag coefficient below 1.0, better than a cube.',
    emoji: '📦',
    passed: (c) => c.snapshot.measurement.cd < 1.0,
  },
  {
    id: 'beat-sphere',
    title: 'Rounder than round',
    hint: 'Get below 0.47, the textbook value for a sphere.',
    emoji: '⚽',
    passed: (c) => c.snapshot.measurement.cd < 0.47,
  },
  {
    id: 'slippery',
    title: 'Slippery',
    hint: 'Get below 0.30. Long, tapered tails help.',
    emoji: '💧',
    passed: (c) => c.snapshot.measurement.cd < 0.3,
  },
  {
    id: 'parachute',
    title: 'Parachute designer',
    hint: 'Go the other way: settle a drag coefficient above 1.3.',
    emoji: '🪂',
    passed: (c) => c.snapshot.measurement.cd > 1.3,
  },
  {
    id: 'lift-off',
    title: 'Lift-off',
    hint: 'Make more lift than drag. Try the wing and tilt its nose up.',
    emoji: '✈️',
    passed: (c) => c.snapshot.measurement.liftToDrag > 1,
  },
  {
    id: 'downforce',
    title: 'Glued to the road',
    hint: 'Produce downforce: a lift coefficient below -0.2.',
    emoji: '🏎️',
    passed: (c) => c.snapshot.measurement.cl < -0.2,
  },
  {
    id: 'rotate',
    title: 'Find the sweet spot',
    hint: 'Cut an object’s drag coefficient by a fifth just by turning it.',
    emoji: '🔄',
    passed: (c) => c.firstCd !== null && c.snapshot.measurement.cd < c.firstCd * 0.8,
  },
  {
    id: 'import',
    title: 'Bring your own',
    hint: 'Import a 3D model of your own and measure it.',
    emoji: '📥',
    passed: (c) => c.imported,
  },
];

const STORAGE_KEY = 'wind-tunnel.challenges';

export function loadCompleted(): Set<string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return new Set(raw ? (JSON.parse(raw) as string[]) : []);
  } catch {
    return new Set();
  }
}

export function saveCompleted(done: Set<string>): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...done]));
  } catch {
    // Storage can be unavailable in private windows; progress then lasts for the session only.
  }
}

/** Returns the challenges newly passed by a settled measurement. */
export function evaluate(context: ChallengeContext, done: Set<string>): Challenge[] {
  if (!context.snapshot.measurement.settled) return [];
  return CHALLENGES.filter((c) => !done.has(c.id) && c.passed(context));
}
