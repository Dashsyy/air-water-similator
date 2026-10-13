/** Commonly quoted drag coefficients, used to put a measured value in context. Approximate by nature. */
export interface ReferenceShape {
  name: string;
  cd: number;
  emoji: string;
}

export const REFERENCE_SHAPES: ReferenceShape[] = [
  { name: 'Streamlined teardrop', cd: 0.04, emoji: '💧' },
  { name: 'Slippery electric car', cd: 0.21, emoji: '⚡' },
  { name: 'Typical family car', cd: 0.3, emoji: '🚗' },
  { name: 'Smooth sphere', cd: 0.47, emoji: '⚽' },
  { name: 'Pickup truck', cd: 0.55, emoji: '🛻' },
  { name: 'Racing cyclist', cd: 0.88, emoji: '🚴' },
  { name: 'Cube', cd: 1.05, emoji: '📦' },
  { name: 'Person standing', cd: 1.15, emoji: '🧍' },
  { name: 'Flat plate, face on', cd: 1.28, emoji: '🧱' },
  { name: 'Parachute', cd: 1.42, emoji: '🪂' },
];

export interface Grade {
  letter: string;
  title: string;
  blurb: string;
}

export function gradeFor(cd: number): Grade {
  if (cd < 0.15) return { letter: 'S', title: 'Slippery as a fish', blurb: 'The air barely notices it. Race-car engineers would like a word.' };
  if (cd < 0.32) return { letter: 'A', title: 'Wind whisperer', blurb: 'As tidy as a modern car. The air flows around it with little fuss.' };
  if (cd < 0.55) return { letter: 'B', title: 'Respectably round', blurb: 'In sphere territory. Not sleek, but nothing to be ashamed of.' };
  if (cd < 0.85) return { letter: 'C', title: 'Bit of a blunt instrument', blurb: 'It leaves a messy wake behind it. Plenty of room to streamline.' };
  if (cd < 1.15) return { letter: 'D', title: 'Honorary cardboard box', blurb: 'About as aerodynamic as a parcel. The wake is doing most of the talking.' };
  return { letter: 'F', title: 'Aerodynamic brick', blurb: 'It fights the air and the air wins. Excellent as a parachute, though.' };
}

/** The reference shape whose drag coefficient is closest on a ratio scale. */
export function closestReference(cd: number): ReferenceShape {
  const value = Math.max(cd, 1e-3);
  let best = REFERENCE_SHAPES[0];
  for (const ref of REFERENCE_SHAPES) {
    if (Math.abs(Math.log(ref.cd / value)) < Math.abs(Math.log(best.cd / value))) best = ref;
  }
  return best;
}
