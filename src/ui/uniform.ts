/**
 * Acapellera's stage: everyone wears the same uniform with a bow tie, drawn in vector lines under the 3D head.
 * Only `color` (the part's group) differs.
 * Sizes are in the body's centimetres at the neck (cmPx); the jacket runs down past the bottom of the screen.
 */
import { beamPts, dot, type Pt } from "./render/vector.ts";
import { V } from "./colors.ts";

type Anchor = { x: number; y: number; cmPx: number };

export function drawUniform(ctx: CanvasRenderingContext2D, color: string, at: { neck: Anchor; h: number }, f: 1 | -1): void {
  const u = at.neck.cmPx, x = at.neck.x + f * 0.8 * u, y = at.neck.y;
  const w = Math.max(1.4, u * 0.16);
  const line = (pts: Pt[], c = color, o = {}) => beamPts(ctx, pts, c, w, o);
  const bottom = at.h + 4;
  // jacket: shoulders out from the collar, then straight down
  for (const s of [-1, 1]) {
    line([[x + s * 3.2 * u, y - 1.2 * u], [x + s * 8 * u, y + 0.6 * u], [x + s * 11.5 * u, y + 2.8 * u], [x + s * 12.5 * u, y + 6 * u], [x + s * 12.8 * u, bottom]]);
    // lapel: from the collar's side down to the V
    line([[x + s * 3.2 * u, y - 1.2 * u], [x + s * 4.4 * u, y + 3.2 * u], [x + s * 2.6 * u, y + 4.2 * u], [x, y + 8.5 * u]]);
    // shirt collar points, white
    line([[x + s * 2.4 * u, y - 1.4 * u], [x + s * 1.8 * u, y + 1.2 * u], [x + s * 0.4 * u, y + 0.4 * u]], V.white, { alpha: 0.9 });
  }
  // the jacket's front edge, and two buttons
  line([[x, y + 8.5 * u], [x, bottom]]);
  dot(ctx, x + 0.9 * u, y + 10.5 * u, Math.max(1.6, 0.45 * u), color);
  dot(ctx, x + 0.9 * u, y + 13.5 * u, Math.max(1.6, 0.45 * u), color);
  // bow tie at the collar
  const by = y + 0.9 * u;
  line([[x, by], [x - 2.8 * u, by - 1.3 * u], [x - 2.8 * u, by + 1.3 * u]], color, { close: true, hot: true });
  line([[x, by], [x + 2.8 * u, by - 1.3 * u], [x + 2.8 * u, by + 1.3 * u]], color, { close: true, hot: true });
  dot(ctx, x, by, Math.max(2, 0.6 * u), color);
}
