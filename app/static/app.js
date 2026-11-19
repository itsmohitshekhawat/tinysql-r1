// TinySQL-R1 demo page: pick a database, ask, and three models answer side by side (streamed as each finishes).
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => Math.round(x * 100) + "%";
const NAMES = { base: "Small model, untrained", sft: "Small model, supervised (sees answers)", grpo: "Small model, GRPO (reward only)",
  sft_grpo: "Small model, supervised then GRPO", deepseek: "Frontier model (DeepSeek)" };
let meta = null, db = "movies", busy = false;

function pickDb(name) {
  db = name;
  document.querySelectorAll("#dbs button").forEach((b) => b.classList.toggle("on", b.dataset.db === name));
  const d = meta.databases[name];
  $("schema").textContent = d.ddl;
  $("schematitle").textContent = `Show the tables of "${name}"` + (d.held_out ? " (never seen in training)" : "");
  $("examples").innerHTML = (meta.examples[name] || []).slice(0, meta.small_live ? 6 : 12).map((e) => `<button class="chip" type="button">${esc(e.question)}</button>`).join("");
}

function card(m) { return document.querySelector(`.model[data-m="${m}"]`); }
function reset() {
  document.querySelectorAll(".model").forEach((c) => {
    c.querySelector(".sql").innerHTML = '<span class="spinner"></span>working…';
    c.querySelector(".sql").classList.add("empty");
    c.querySelector(".out").innerHTML = ""; c.querySelector(".stats").innerHTML = "";
    const t = c.querySelector(".tag"); t.hidden = true; t.className = "tag";
  });
}

function show(r) {
  const c = card(r.model); if (!c) return;
  const sql = c.querySelector(".sql"), out = c.querySelector(".out");
  sql.classList.remove("empty");
  if (r.failed) { sql.textContent = "This model did not answer (" + r.failed + ")."; return; }
  if (r.offline) { sql.textContent = "The small model is not running live on this page. Pick one of the test questions above to see its recorded answer."; return; }
  sql.textContent = r.sql || "(no SQL in the reply)";
  if (r.error) out.innerHTML = `<p class="err">Does not run: ${esc(r.error)}</p>`;
  else if (!r.row_count) out.innerHTML = '<p class="hint">Runs, but returns no rows.</p>';
  else out.innerHTML = `<div class="scroll"><table class="rows">${r.rows.map((row) => "<tr>" + row.map((v) => `<td>${esc(v === null ? "null" : v)}</td>`).join("") + "</tr>").join("")}</table></div>` +
    (r.row_count > r.rows.length ? `<p class="hint">${r.row_count} rows, first ${r.rows.length} shown</p>` : "");
  const t = c.querySelector(".tag");
  if (r.correct !== undefined) { t.hidden = false; t.textContent = r.correct ? "correct" : "wrong"; t.classList.add(r.correct ? "ok" : "no"); }
  c.querySelector(".stats").innerHTML = r.saved ? '<span class="stat">answer recorded in the benchmark run</span>'
    : `<span class="stat">${r.seconds}s</span><span class="stat">${r.cost_usd ? "$" + r.cost_usd.toFixed(5) : "$0 (own CPU)"}</span>`;
}

async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true; $("go").disabled = true; $("q").value = question; reset();
  try {
    const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ db, question }) });
    if (!res.ok || !res.body) throw new Error("http " + res.status);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i); buf = buf.slice(i + 2);
        const ev = (block.match(/^event: (.*)$/m) || [])[1], data = (block.match(/^data: (.*)$/m) || [])[1];
        if (ev === "result") show(JSON.parse(data));
        if (ev === "limit" || ev === "failed") document.querySelectorAll(".model .sql").forEach((s) => (s.textContent = ev === "limit" ? "The free daily limit is reached. The results below are still available." : "Something went wrong. Please try again."));
      }
    }
  } catch {
    document.querySelectorAll(".model .sql").forEach((s) => { if (s.querySelector(".spinner")) s.textContent = "Connection problem. Please try again."; });
  }
  document.querySelectorAll(".model .sql .spinner").forEach((s) => (s.parentElement.textContent = "No answer in time."));
  busy = false; $("go").disabled = false; quota();
}

async function quota() {
  try {
    const s = await (await fetch("/api/status")).json();
    $("quota").textContent = s.open ? `${s.your_questions_left} questions left for you today. A test question shows correct / wrong marks.` : "Daily free limit reached.";
  } catch { /* ignore */ }
}

function bars(el, results, key) {
  const order = ["base", "sft", "grpo", "sft_grpo", "deepseek"].filter((n) => results[n]);
  el.innerHTML = order.map((n) => `<div class="barrow"><span>${NAMES[n]}</span><div class="bar"><i style="width:${results[n][key] * 100}%"></i></div><span>${pct(results[n][key])}</span></div>`).join("");
}

function curve(training) {
  if (!training || !training.steps || !training.steps.length) { $("curve").innerHTML = '<p class="hint">Training log not available.</p>'; return; }
  const W = 720, H = 240, L = 40, R = 12, T = 12, B = 28, steps = training.steps, n = steps[steps.length - 1].step;
  const x = (s) => L + (W - L - R) * (s / n), y = (v) => T + (H - T - B) * (1 - v);
  const win = 25, pts = steps.map((e, i) => { const a = steps.slice(Math.max(0, i - win + 1), i + 1); return [e.step, a.reduce((t, z) => t + z.correct, 0) / a.length]; });
  const line = pts.map(([s, v], i) => (i ? "L" : "M") + x(s).toFixed(1) + " " + y(v).toFixed(1)).join(" ");
  const grid = [0, 0.25, 0.5, 0.75, 1].map((v) => `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${pct(v)}</text>`).join("");
  const dots = (training.dev || []).map((d) => `<circle cx="${x(d.step)}" cy="${y(d.accuracy)}" r="4" fill="var(--panel)" stroke="var(--accent)" stroke-width="2"><title>step ${d.step}: ${pct(d.accuracy)}</title></circle>`).join("");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => `<text x="${x(n * f)}" y="${H - 8}" text-anchor="middle">${Math.round(n * f)}</text>`).join("");
  $("curve").innerHTML = `<svg class="curve" viewBox="0 0 ${W} ${H}" role="img" aria-label="Training curve">${grid}${ticks}<path d="${line}" fill="none" stroke="var(--accent)" stroke-width="2"/>${dots}</svg>`;
}

fetch("/api/meta").then((r) => r.json()).then((d) => {
  meta = d;
  const names = Object.keys(d.databases).sort((a, b) => d.databases[b].held_out - d.databases[a].held_out);
  $("dbs").innerHTML = names.map((n) => `<button type="button" data-db="${n}">${n}${d.databases[n].held_out ? "<small>unseen</small>" : ""}</button>`).join("");
  $("dbs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) pickDb(b.dataset.db); });
  $("examples").addEventListener("click", (e) => { const b = e.target.closest(".chip"); if (b) ask(b.textContent); });
  pickDb(names.includes("movies") ? "movies" : names[0]);
  $("livenote").textContent = d.small_live ? "The small model runs live on this server's CPU."
    : "On this page the small model's answers to the test questions below are the ones recorded in the benchmark run; the frontier model answers live.";
  if (!d.small_live) $("q").placeholder = "Pick a test question below, or type your own (frontier model only)";
  const r = d.results;
  if (r.base) {
    $("resultsintro").textContent = `Execution accuracy: the share of test questions where the model's query returns the right rows. ${d.counts.test_seen} new questions on the 6 training databases, ${d.counts.test_unseen} questions on 2 databases the model never saw. Trained on ${d.counts.train} questions.`;
    bars($("bars-seen"), r, "test_seen"); bars($("bars-unseen"), r, "test_unseen");
    const order = ["base", "sft", "grpo", "sft_grpo", "deepseek"].filter((n) => r[n]);
    $("table").innerHTML = "<tr><th>Model</th><th>All</th><th>Easy</th><th>Medium</th><th>Hard</th><th>Query runs</th><th>Seconds / question</th><th>Cost / question</th></tr>" +
      order.map((n) => `<tr class="${n === "grpo" ? "best" : ""}"><td>${NAMES[n]}</td><td>${pct(r[n].all)}</td><td>${pct(r[n].all_easy)}</td><td>${pct(r[n].all_medium)}</td><td>${pct(r[n].all_hard)}</td><td>${pct(r[n].runs_without_error)}</td><td>${r[n].seconds_per_question}</td><td>${r[n].cost_per_question_usd ? "$" + r[n].cost_per_question_usd.toFixed(5) : "$0"}</td></tr>`).join("");
    const items = [];
    if (r.grpo) items.push(`Reinforcement learning with a checkable reward took the small model from ${pct(r.base.all)} to ${pct(r.grpo.all)} overall, without ever showing it a correct query.`);
    if (r.grpo) items.push(`On databases it never saw, accuracy went from ${pct(r.base.test_unseen)} to ${pct(r.grpo.test_unseen)}: the gain is not just memorised schemas.`);
    if (r.grpo) items.push(`The gain is on easy questions (${pct(r.base.all_easy)} to ${pct(r.grpo.all_easy)}). Hard questions got worse (${pct(r.base.all_hard)} to ${pct(r.grpo.all_hard)}), and the model picked up a lazy habit: ${pct(r.grpo.select_star)} of its answers use SELECT * (untrained: ${pct(r.base.select_star)}). Extra columns are allowed, so this is not cheating, but it is a style a reviewer would reject.`);
    if (r.sft && r.grpo) items.push(`Supervised training, which does see the correct queries, is much better: ${pct(r.sft.all)}. With 455 verified examples available, showing answers beats rewarding them. GRPO matters when you can check answers but cannot write them.`);
    items.push("Training was unstable: development accuracy peaked at step 50 and then fell; the saved model is the best checkpoint on the development set, chosen before the test sets were run.");
    if (r.deepseek && r.grpo) items.push(`The frontier model is far better (${pct(r.deepseek.all)}), especially on hard questions (${pct(r.deepseek.all_hard)}). The small model is 0.5B parameters; the point is the method and the honest measurement, not beating it.`);
    items.push("Limits: the questions were written by a teacher model and verified by running them; the test sets are small (175 questions); scores come from one training run.");
    $("takeaways").innerHTML = items.map((x) => `<li>${x}</li>`).join("");
  }
  curve(d.training);
  const h = d.reward_hacking;
  if (h) {
    $("hacked").hidden = false;
    $("hackintro").textContent = `My first training run went from ${pct(h.lenient.base)} to ${pct(h.lenient.grpo)} on the test questions. Then I read its answers. It had not learned SQL; it had learned a hole in my scoring. Scored fairly, the same model was at ${pct(h.fair.grpo)}: no gain at all.`;
    $("hackq").textContent = "Question: " + h.example.question;
    $("hacksql").textContent = h.example.model;
    const row = (label, v) => `<div class="barrow"><span>${label}</span><div class="bar"><i style="width:${v * 100}%"></i></div><span>${pct(v)}</span></div>`;
    $("hackbars").innerHTML = row("Untrained, first rule", h.lenient.base) + row("Trained, first rule", h.lenient.grpo) + row("Untrained, fixed rule", h.fair.base) + row("Trained, fixed rule", h.fair.grpo);
  }
});
$("form").addEventListener("submit", (e) => { e.preventDefault(); ask($("q").value); });
quota();
