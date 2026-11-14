/**
 * Acapellera's stage (`?stage=<port of acapellera/drop_server.py>`), opened by acapellera/launch.py. One page
 * from the dropped song to the bow:
 *
 *   drop a song  -> the curtain is down; the % and the step being done show in front of it, and the
 *                   audience murmur (hall.mp3, optional) grows from silence with the %
 *   show is made -> the buzzer; the murmur fades away slowly; Play waits
 *   Play         -> the curtain sound (optional) and the curtain parts, then three light sounds one after the other; with each
 *                   the back row (chorus squad and drums), the harmony, then the lead appear (no light cones drawn),
 *                   then the song starts
 *   song ends    -> the singers stay, the curtain closes to its sound played backwards, then another drop
 *                   or the download
 *
 * Rows: the chorus squad and the drums at the back; the harmony and, largest in the middle, the lead at the
 * front. The lead grows where the song lifts. Everyone wears the same uniform with a bow tie (uniform.ts), coloured only by group. Download: the mix as wav.
 */
import { useEffect, useRef, useState } from "react";
import { Transport } from "../../show/transport.ts";
import { Head, loadShow, clock, type Loaded } from "./DuetPage.tsx";
import { drawUniform } from "../uniform.ts";
import { V } from "../colors.ts";
import "./stage.css";

// the sounds (public/stage-sfx/, see its CREDITS.md): a missing one is skipped, the stage goes on without it
const SFX = `${import.meta.env.BASE_URL}stage-sfx/`;
const sfx = (p: string) => SFX + p.split("/").map(encodeURIComponent).join("/");
// the shortest takes of each sound, so the curtain-up takes about 7 s (lengths measured 2026-11-12)
const HALL = sfx("hall.mp3"); // 37.06 s, looped
const BUZZER = { src: sfx("Opening_Buzzer01-1.mp3"), s: 3.81 };
// one light sound per cue: back row, harmony, lead
const SPOTS = [sfx("Spotlight03-3.mp3"), sfx("Spotlight02-3.mp3"), sfx("Spotlight05-2(Short).mp3")];
// the buzzer when Play can be pressed; Play: the curtain sound and the curtain opens; at the end it closes to the
// same sound played backwards
const CURTAIN = { src: sfx("curtain03.mp3"), s: 1.7 }; // 1.70 s (measured); the curtain moves for as long
const SPOT_EVERY = 1.2; // s between the lights
const SONG_AFTER = 1.0; // s from the last light to the song

/** voice order in show.json (make_sing.py): see PARTS. One uniform for all; the colour
 *  only by group (vocal, harmony, the rest); the colours are my choice. */
const GROUP = { vocal: V.amber, harmony: V.green, drums: V.pink, chorus: V.white };
// `at`: the place in its row, left to right
// show.json's order (make_sing.py): lead, do, ree, bah, pa, si, then the chorus squad zoo, di, no, ku, ru, mi
// Two rows: the back row zoo di no ku bah pa si ru mi, the front row do, lead, ree (the lead largest).
// `light` is the cue (1-3) with which the part appears: the back row, then the harmony, then the lead.
const PARTS: { color: string; row: 0 | 1; at: number; light: 1 | 2 | 3 }[] = [
  { color: GROUP.vocal, row: 1, at: 1, light: 3 },
  { color: GROUP.harmony, row: 1, at: 0, light: 2 },
  { color: GROUP.harmony, row: 1, at: 2, light: 2 },
  { color: GROUP.drums, row: 0, at: 4, light: 1 },
  { color: GROUP.drums, row: 0, at: 5, light: 1 },
  { color: GROUP.drums, row: 0, at: 6, light: 1 },
  // the chorus squad in the drums' back row, in white: bass zoo, di and guitar no, ku on the left of the drums,
  // piano ru, mi on the right
  { color: GROUP.chorus, row: 0, at: 0, light: 1 },
  { color: GROUP.chorus, row: 0, at: 1, light: 1 },
  { color: GROUP.chorus, row: 0, at: 2, light: 1 },
  { color: GROUP.chorus, row: 0, at: 3, light: 1 },
  { color: GROUP.chorus, row: 0, at: 7, light: 1 },
  { color: GROUP.chorus, row: 0, at: 8, light: 1 },
];
// the camera pulled back and aimed lower so the uniform shows under the head (checked on the stage pictures)
const ZOOM_OUT = 1.45;
const LOWER_CM = 8;
const ROWS = ["back", "front"] as const;

// The curtain in the heads' line style, in two halves that sway a little while closed: each half is fold lines, each line a chain of masses
// hanging from the rail (a small mass-spring simulation of my own, not a library): a spring back to its rest place,
// tension to its neighbours, damping, and a slow wind; the bottom swings most. Stops while the curtain is open.
const FOLDS_PER_HALF = 11;
const LINKS = 24;
function CurtainHalf(props: { side: "left" | "right"; open: boolean }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const openRef = useRef(props.open);
  openRef.current = props.open;
  useEffect(() => {
    const cv = ref.current!;
    const ctx = cv.getContext("2d")!;
    // chains 0..FOLDS_PER_HALF-1 are the folds; the last one is the inner edge where the two halves meet, which sways like a fold instead of standing as a ruled line
    const dx = Array.from({ length: FOLDS_PER_HALF + 1 }, () => new Float32Array(LINKS)); // sideways offset (px)
    const vx = Array.from({ length: FOLDS_PER_HALF + 1 }, () => new Float32Array(LINKS));
    let raf = 0, last = performance.now(), t = 0;
    const seed = props.side === "left" ? 0 : 5.3;
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (openRef.current) return; // nothing moves (and nothing is computed) while it is open
      t += dt;
      const dpr = Math.min(1.5, window.devicePixelRatio || 1);
      const w = cv.clientWidth, h = cv.clientHeight;
      if (cv.width !== Math.round(w * dpr)) cv.width = Math.round(w * dpr);
      if (cv.height !== Math.round(h * dpr)) cv.height = Math.round(h * dpr);
      // the physics, in a few small steps per frame
      for (let s = 0; s < 3; s++) {
        const h3 = dt / 3;
        for (let k = 0; k <= FOLDS_PER_HALF; k++) {
          const x = dx[k], v = vx[k];
          for (let i = 1; i < LINKS; i++) {
            const down = i / (LINKS - 1);
            const wind = (5 + 4 * Math.sin(0.37 * t + seed)) * Math.sin(0.9 * t + 0.6 * k + seed + 2.2 * down) * down;
            const tension = 26 * ((x[i - 1] - x[i]) + (i + 1 < LINKS ? x[i + 1] - x[i] : 0));
            const a = wind * 6 + tension - 3.5 * x[i] - 1.6 * v[i];
            v[i] += a * h3;
          }
          for (let i = 1; i < LINKS; i++) x[i] += v[i] * h3;
        }
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const gap = w / FOLDS_PER_HALF;
      const left = props.side === "left";
      // the inner edge (the seam): a hanging chain with the folds' wave and sway, a little inside the canvas
      // (at 0.12 gap it was cut straight at the canvas edge)
      const edgeX = left ? w - 0.42 * gap : 0.42 * gap;
      const edgeBottom = h * 0.975 - (dx[FOLDS_PER_HALF][LINKS - 1] ** 2) / (0.6 * h);
      const edge: [number, number][] = [];
      for (let i = 0; i < LINKS; i++) {
        // the folds' own wave and sway: period 0.9 a link, 0.18 of the fold spacing
        edge.push([edgeX + 0.18 * gap * Math.sin(i * 0.9 + FOLDS_PER_HALF * 1.7) + dx[FOLDS_PER_HALF][i], (edgeBottom * i) / (LINKS - 1)]);
      }
      // the cloth: dark only out from the seam, so the half behind is hidden exactly along the swaying seam, not
      // along the canvas's straight edge (else the middle line breaks as it sways); folds are kept inside it too
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(left ? 0 : w, 0);
      edge.forEach(([px, py]) => ctx.lineTo(px, py));
      ctx.lineTo(edge[LINKS - 1][0], h);
      ctx.lineTo(left ? 0 : w, h);
      ctx.closePath();
      ctx.fillStyle = "rgba(0, 0, 0, 0.94)";
      ctx.fill();
      ctx.clip();
      ctx.lineWidth = 1.4;
      ctx.shadowBlur = 6;
      ctx.shadowColor = "rgba(255, 93, 64, 0.85)";
      ctx.strokeStyle = "#ff5d40";
      const ends: [number, number][] = [];
      for (let k = 0; k < FOLDS_PER_HALF; k++) {
        const x0 = gap * (k + 0.5);
        // fabric, not a ruler (a straight bottom edge looked ruled): the folds that
        // come forward hang lower than the ones behind, and a fold swung aside lifts its end a little
        const swing = dx[k][LINKS - 1];
        const bottom = h * (k % 2 === 0 ? 0.97 : 0.94) - (swing * swing) / (0.6 * h); // the sag below stays on the canvas
        ctx.beginPath();
        for (let i = 0; i < LINKS; i++) {
          const y = (bottom * i) / (LINKS - 1);
          const px = x0 + 0.18 * gap * Math.sin(i * 0.9 + k * 1.7) + dx[k][i];
          if (i) ctx.lineTo(px, y);
          else ctx.moveTo(px, y);
          if (i === LINKS - 1) ends.push([px, y]);
        }
        ctx.stroke();
      }
      ctx.restore();
      // the seam line: the same line as the red folds, only orange
      ctx.lineWidth = 1.4;
      ctx.shadowBlur = 6;
      ctx.strokeStyle = "#ff8a3d";
      ctx.shadowColor = "rgba(255, 138, 61, 0.85)";
      ctx.beginPath();
      edge.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
      ctx.stroke();
      const edgeEnd = edge[LINKS - 1];
      // the hem, amber: from the wall through every fold's end to the inner edge's end, sagging between them like cloth
      const pts = [...ends, edgeEnd].sort((a, b) => a[0] - b[0]);
      ctx.strokeStyle = "#ffb340";
      ctx.shadowColor = "rgba(255, 179, 64, 0.8)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      const sag = 0.018 * h;
      if (left) {
        ctx.moveTo(0, pts[0][1] - sag);
        ctx.quadraticCurveTo(pts[0][0] / 2, pts[0][1] + sag, pts[0][0], pts[0][1]);
      } else ctx.moveTo(pts[0][0], pts[0][1]);
      for (let k = 1; k < pts.length; k++) {
        const [ax, ay] = pts[k - 1], [bx, by] = pts[k];
        ctx.quadraticCurveTo((ax + bx) / 2, Math.max(ay, by) + sag, bx, by);
      }
      if (!left) {
        const [lx, ly] = pts[pts.length - 1];
        ctx.quadraticCurveTo((lx + w) / 2, ly + sag, w, ly - sag);
      }
      ctx.stroke();
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [props.side]);
  return <canvas ref={ref} className={`curtain-half ${props.side}`} />;
}

const VALANCE = "M0,0 L0,6 " + Array.from({ length: 20 }, (_, k) => `Q${k * 5 + 2.5},10 ${k * 5 + 5},6`).join(" ") + " L100,0";

interface Status {
  lines: string[];
  done: boolean;
  show: string | null;
  pct: number;
  stage: string;
  wav: boolean;
  /** what went wrong and what to do, in plain words (drop_server.plain) */
  error: string | null;
  wav_failed: boolean;
  notices: string[];
  /** made, but without something (e.g. the sections), in plain words */
}

// The lead's size follows its voice (make_sing.LIFT_DB: chorus +3 dB, pre-chorus +1 dB)
const LIFT_SCALE: Record<string, number> = { chorus: 1.14, "pre-chorus": 1.05, prechorus: 1.05 };

type Phase = "empty" | "making" | "ready" | "opening" | "open" | "closing" | "ended";

/**
 * The audience murmur, looped without a gap: each pass of the recording fades in over its first XFADE s and out over its last XFADE s,
 * and the next pass starts XFADE before the end of the one playing, so the two overlap there. `volume` 0..1.
 */
const XFADE = 0.1;
const MURMUR_MAX = 0.35; // the murmur's level at 100 % (kept low)
const MURMUR_FADE = 8; // s, the murmur dying away slowly after the buzzer
class Murmur {
  private out: GainNode;
  private buf: AudioBuffer | null = null;
  private next = 0;
  private timer = 0;
  paused = true;
  constructor(private ctx: AudioContext, src: string) {
    this.out = ctx.createGain();
    this.out.gain.value = 0;
    this.out.connect(ctx.destination);
    void fetch(src)
      .then((r) => r.arrayBuffer())
      .then((b) => ctx.decodeAudioData(b))
      .then((b) => {
        this.buf = b;
        if (!this.paused) this.schedule();
      })
      .catch(() => {}); // no murmur file (it is optional): the stage goes on silent
  }
  get volume(): number {
    return this.out.gain.value;
  }
  set volume(v: number) {
    this.out.gain.setTargetAtTime(v, this.ctx.currentTime, 0.15);
  }
  start(): void {
    this.paused = false;
    this.next = this.ctx.currentTime + 0.05;
    if (this.buf) this.schedule();
  }
  /** fade to nothing over `s` seconds, then stop */
  stop(s: number): void {
    this.out.gain.setTargetAtTime(0, this.ctx.currentTime, s / 4);
    window.setTimeout(() => {
      this.paused = true;
      clearTimeout(this.timer);
    }, s * 1000);
  }
  private schedule(): void {
    if (this.paused || !this.buf) return;
    const d = this.buf.duration;
    while (this.next < this.ctx.currentTime + 2) {
      const src = this.ctx.createBufferSource();
      src.buffer = this.buf;
      const g = this.ctx.createGain();
      const t = this.next;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(1, t + XFADE);
      g.gain.setValueAtTime(1, t + d - XFADE);
      g.gain.linearRampToValueAtTime(0, t + d);
      src.connect(g).connect(this.out);
      src.start(t);
      this.next = t + d - XFADE;
    }
    this.timer = window.setTimeout(() => this.schedule(), 1000);
  }
}

/** a sound played backwards (the curtain closing is its opening sound reversed) */
async function playReversed(ctx: AudioContext, src: string): Promise<void> {
  const buf = await ctx.decodeAudioData(await (await fetch(src)).arrayBuffer());
  for (let c = 0; c < buf.numberOfChannels; c++) buf.getChannelData(c).reverse();
  const s = ctx.createBufferSource();
  s.buffer = buf;
  s.connect(ctx.destination);
  s.start();
}

function play(src: string, volume = 1): HTMLAudioElement {
  const a = new Audio(src);
  a.volume = volume;
  void a.play().catch(() => {});
  return a;
}

/** the app's name */
const APP = "Acapellera";

export function StagePage(props: { port: number }) {
  const api = `http://127.0.0.1:${props.port}`;
  useEffect(() => {
    document.title = APP;
  }, []);
  const [phase, setPhase] = useState<Phase>("empty");
  const [status, setStatus] = useState<Status | null>(null);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [lit, setLit] = useState(0); // rows lit, from the back
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const [transport, setTransport] = useState<Transport | null>(null);
  const [, setTick] = useState(0);
  const hall = useRef<Murmur | null>(null);
  const timers = useRef<number[]>([]);

  useEffect(() => {
    const t = new Transport();
    setTransport(t);
    const off = t.subscribe(() => setTick((n) => n + 1));
    const id = setInterval(() => t.isPlaying && setTick((n) => n + 1), 250);
    return () => {
      off();
      clearInterval(id);
      t.dispose();
    };
  }, []);

  // while the show is made: ask the server how far it is, twice a second; the murmur follows the %
  useEffect(() => {
    if (phase !== "making" || !transport) return;
    let live = true;
    const id = setInterval(async () => {
      try {
        const s = (await (await fetch(api + "/status")).json()) as Status;
        if (!live) return;
        setStatus(s);
        if (hall.current) hall.current.volume = MURMUR_MAX * Math.min(1, s.pct / 100);
        if (s.done) {
          clearInterval(id);
          if (!s.show) {
            // the curtain stays down, the murmur dies away, and the next drop is taken
            hall.current?.stop(1.5);
            setPhase("empty");
            return setError(s.error ?? "途中で止まりました。もう一度ドロップしてください。");
          }
          let l: Loaded;
          try {
            l = await loadShow(`${import.meta.env.BASE_URL}shows/${s.show}/`, transport);
          } catch {
            if (!live) return;
            hall.current?.stop(1.5);
            setPhase("empty");
            return setError("できた歌を舞台に並べられませんでした。もう一度ドロップしてください。");
          }
          if (live) {
            setLoaded(l);
            setPhase("ready");
            play(BUZZER.src); // Play can be pressed now
            hall.current?.stop(MURMUR_FADE); // the murmur fades out slowly once the buzzer sounds
          }
        }
      } catch {
        if (live) {
          clearInterval(id);
          hall.current?.stop(1.5);
          setPhase("empty");
          setError("処理する側との接続が切れました。起動した窓が動いているか確かめて、もう一度ドロップしてください。");
        }
      }
    }, 500);
    return () => {
      live = false;
      clearInterval(id);
    };
  }, [phase, transport, api]);

  // after the show: keep asking until the wav for the download has arrived
  useEffect(() => {
    if (phase === "empty" || phase === "making" || status?.wav || status?.wav_failed) return;
    const id = setInterval(async () => {
      try {
        const s = (await (await fetch(api + "/status")).json()) as Status;
        if (s.wav || s.wav_failed) setStatus(s);
      } catch {
        /* the server is gone: nothing to download */
      }
    }, 1000);
    return () => clearInterval(id);
  }, [phase, status?.wav, api]);

  const clearTimers = () => {
    timers.current.forEach((t) => clearTimeout(t));
    timers.current = [];
  };

  async function drop(file: File) {
    if (phase === "making") return;
    clearTimers();
    transport?.pause();
    setLoaded(null);
    setLit(0);
    setError(null);
    setStatus(null);
    if (!transport) return;
    // the drop is a user gesture: wake the audio context now, for the murmur and for Play later
    void transport.ctx.resume();
    // the murmur starts from silence with the drop and grows with the %
    hall.current?.stop(0.3);
    hall.current = new Murmur(transport.ctx, HALL);
    hall.current.start();
    (window as unknown as { __stageHall?: Murmur }).__stageHall = hall.current; // read by web/check_stage.ts
    try {
      const r = await fetch(api + "/upload", { method: "POST", headers: { "X-Name": encodeURIComponent(file.name) }, body: file });
      if (!r.ok) return setError(await r.text());
      setPhase("making");
    } catch (e) {
      setError(`送信できませんでした（起動した窓は動いていますか）: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  function curtainUp() {
    if (!transport || !loaded) return;
    setPhase("opening");
    // the murmur dies down under the buzzer
    hall.current?.stop(2.5);
    play(CURTAIN.src);
    SPOTS.forEach((s, i) =>
      timers.current.push(
        window.setTimeout(() => {
          play(s);
          setLit(i + 1);
        }, (CURTAIN.s + i * SPOT_EVERY) * 1000),
      ),
    );
    timers.current.push(
      window.setTimeout(() => {
        setPhase("open");
        transport.seek(0);
        transport.play();
      }, (CURTAIN.s + (SPOTS.length - 1) * SPOT_EVERY + SONG_AFTER) * 1000),
    );
  }

  // the song has ended: the curtain closes over the singers to its opening sound played backwards, and the
  // stage offers another drop or the download
  useEffect(() => {
    if (phase !== "open") return;
    const id = setInterval(() => setTick((n) => n + 1), 300); // so the end is seen even when nothing else redraws
    return () => clearInterval(id);
  }, [phase]);
  const ended = phase === "open" && transport !== null && !transport.isPlaying && transport.duration > 0 && transport.time() >= transport.duration - 0.05;
  useEffect(() => {
    if (!ended || !transport) return;
    // the singers stay as they are while the curtain closes, and go only once it hides them
    setPhase("closing");
    void playReversed(transport.ctx, CURTAIN.src).catch(() => {}); // no curtain sound file: closes silently
    timers.current.push(
      window.setTimeout(() => {
        setPhase("ended");
        setLit(0);
      }, CURTAIN.s * 1000),
    );
  }, [ended, transport]);

  const st = transport?.state ?? { playing: false, time: 0, duration: 0, epoch: 0 };
  const curtainOpen = phase === "opening" || phase === "open"; // closing and ended: down again
  const title = loaded?.show.title ?? "";
  const pct = phase === "making" ? (status?.pct ?? 0) : 100;
  // the lead grows where the song lifts, by the section's own label, as its voice does
  const section = phase === "open" ? (loaded?.show.sections ?? []).find((s) => st.time >= s.start && st.time < s.end) : undefined;
  const leadScale = LIFT_SCALE[section?.label.toLowerCase() ?? ""] ?? 1;

  return (
    <div
      className={`stage${over ? " drag" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const f = e.dataTransfer.files[0];
        if (f) void drop(f);
      }}
    >
      <header className="duet-top">
        <span className="wordmark">{APP}</span>
        <span className="duet-title">{title}</span>
      </header>
      <main className="stage-floor">
        {loaded && transport && (
          <div className="stage-rows">
            {ROWS.map((name, row) => {
              const members = loaded.voices
                .map((v, i) => ({ v, i }))
                .filter(({ i }) => PARTS[i]?.row === row)
                .sort((a, b) => PARTS[a.i].at - PARTS[b.i].at);
              return (
                <div key={name} className={`stage-row stage-${name}${members.some(({ i }) => lit >= PARTS[i].light) ? " lit" : ""}`} style={row === 1 ? ({ "--lead-scale": leadScale } as React.CSSProperties) : undefined}>
                  {members.map(({ v, i }, k) => (
                    <Head
                      key={i}
                      className={`duet-head stage-head${i === 0 ? " stage-lead" : ""}${lit >= PARTS[i].light ? " lit" : ""}`}
                      name={v.name.replace(/ \(.*\)$/, "")} /* the syllable only: nine long names overlapped in the back row */
                      track={v.track}
                      words={v.words}
                      show={loaded.show}
                      transport={transport}
                      facing={members.length > 1 && k === members.length - 1 ? -1 : 1}
                      color={PARTS[i].color}
                      decor={(ctx, at, f) => drawUniform(ctx, PARTS[i].color, at, f)}
                      zoomOut={ZOOM_OUT}
                      lower={LOWER_CM}
                    />
                  ))}
                </div>
              );
            })}
          </div>
        )}
        <div className={`curtain${curtainOpen ? " open" : ""}`} aria-hidden="true">
          {(["left", "right"] as const).map((side) => (
            <CurtainHalf key={side} side={side} open={curtainOpen} />
          ))}
          <svg className="curtain-valance" viewBox="0 0 100 10" preserveAspectRatio="none">
            <path d={VALANCE} />
          </svg>
        </div>
        {!curtainOpen && (
          <div className="stage-panel">
            {phase === "empty" && <p className="stage-call">曲のファイルをここにドロップ</p>}
            {phase === "ended" && (
              <>
                <p className="stage-call">別の曲をここにドロップ</p>
                {status?.wav && status.show ? (
                  <a className="stage-download big" href={`${import.meta.env.BASE_URL}shows/${status.show}/mix.wav`} download={`${title.replace(/, a cappella$/, "") || status.show} (Acapellera).wav`}>
                    今の歌をダウンロード (.wav)
                  </a>
                ) : (
                  <span className="stage-download off">{status?.wav_failed ? "wav を用意できませんでした" : "wav 準備中"}</span>
                )}
              </>
            )}
            {phase === "making" && (
              <>
                <p className="stage-step">
                  <span className="stage-pct">{Math.floor(pct)}%</span>
                </p>
                <div className="stage-bar" role="progressbar" aria-valuenow={Math.floor(pct)} aria-valuemin={0} aria-valuemax={100}>
                  <div style={{ width: `${pct}%` }} />
                </div>
              </>
            )}
            {/* Play alone, large, nothing under it; no notes on the page at all */}
            {phase === "ready" && (
              <button className="primary large stage-play" onClick={curtainUp}>
                Play
              </button>
            )}
            {error && <p className="stage-error">{error}</p>}
            {status && (phase === "making" || error) && <pre className="stage-log">{status.lines.slice(-4).join("\n")}</pre>}
          </div>
        )}
      </main>
      <footer className="duet-bar">
        <button className="primary large" disabled={phase !== "open"} onClick={() => (st.playing ? transport?.pause() : transport?.play())}>
          {st.playing ? "Pause" : st.duration > 0 && st.time >= st.duration - 0.05 ? "Again" : "Play"}
        </button>
        <span className="duet-time">
          {clock(st.time)} / {clock(st.duration)}
        </span>
        {status?.wav && status.show ? (
          <a className="stage-download" href={`${import.meta.env.BASE_URL}shows/${status.show}/mix.wav`} download={`${title.replace(/, a cappella$/, "") || status.show} (Acapellera).wav`}>
            Download .wav
          </a>
        ) : (
          <span className="stage-download off">{phase === "empty" || phase === "making" ? "" : status?.wav_failed ? "wav を用意できませんでした" : "wav 準備中"}</span>
        )}
      </footer>
    </div>
  );
}
