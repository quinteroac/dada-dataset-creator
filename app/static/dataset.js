const panel = document.querySelector("[data-dataset-slug]");
const jobsList = document.querySelector("#jobs-list");
const initialJobs = window.__initialJobs || [];
const completedKey = panel ? `dada-completed-jobs-${panel.dataset.datasetSlug}` : "";
const completedJobs = new Set(readCompletedJobs());
const qwenVramPreset = document.querySelector("#qwen-vram-preset");
const qwenBlocksToSwap = document.querySelector("#qwen-blocks-to-swap");
const qwenNetworkDim = document.querySelector('input[name="network_dim"]');
const qwenTrainingBackend = document.querySelector("#qwen-training-backend");
const modalTrainingOptions = document.querySelector(".modal-training-options");
const imageSelectionBoxes = [...document.querySelectorAll("[data-image-select]")];
const selectedImageCount = document.querySelector("[data-selected-image-count]");
const deleteSelectedImages = document.querySelector("[data-delete-selected-images]");
const selectAllImages = document.querySelector("[data-select-all-images]");
const clearImageSelection = document.querySelector("[data-clear-image-selection]");
const bulkDeleteForm = document.querySelector("#bulk-delete-form");

function readCompletedJobs() {
  if (!completedKey) return [];
  try {
    const value = JSON.parse(localStorage.getItem(completedKey) || "[]");
    return Array.isArray(value) ? value : [];
  } catch {
    localStorage.removeItem(completedKey);
    return [];
  }
}

if (qwenVramPreset && qwenBlocksToSwap && qwenNetworkDim) {
  qwenVramPreset.addEventListener("change", () => {
    const option = qwenVramPreset.selectedOptions[0];
    qwenBlocksToSwap.value = option.dataset.blocks || qwenBlocksToSwap.value;
    qwenNetworkDim.value = option.dataset.dim || qwenNetworkDim.value;
  });
}

function syncTrainingBackendOptions() {
  if (!qwenTrainingBackend || !modalTrainingOptions) return;
  modalTrainingOptions.hidden = qwenTrainingBackend.value !== "modal";
}

if (qwenTrainingBackend) {
  qwenTrainingBackend.addEventListener("change", syncTrainingBackendOptions);
  syncTrainingBackendOptions();
}

function syncImageSelection() {
  if (!selectedImageCount || !deleteSelectedImages) return;
  const selected = imageSelectionBoxes.filter((box) => box.checked).length;
  selectedImageCount.textContent = selected === 1 ? "1 selected" : `${selected} selected`;
  deleteSelectedImages.disabled = selected === 0;
}

document.addEventListener("change", (event) => {
  if (event.target.matches("[data-image-select]")) {
    syncImageSelection();
  }
});

if (selectAllImages) {
  selectAllImages.addEventListener("click", () => {
    imageSelectionBoxes.forEach((box) => {
      box.checked = true;
    });
    syncImageSelection();
  });
}

if (clearImageSelection) {
  clearImageSelection.addEventListener("click", () => {
    imageSelectionBoxes.forEach((box) => {
      box.checked = false;
    });
    syncImageSelection();
  });
}

if (bulkDeleteForm) {
  bulkDeleteForm.addEventListener("submit", (event) => {
    const selected = imageSelectionBoxes.filter((box) => box.checked).length;
    if (!selected || !confirm(`Delete ${selected} selected image${selected === 1 ? "" : "s"} from this dataset?`)) {
      event.preventDefault();
    }
  });
  syncImageSelection();
}

function statusLabel(status) {
  return `<span class="job-status ${status}">${status}</span>`;
}

function formatDate(value) {
  if (!value) return "";
  try {
    return new Date(value).toLocaleTimeString();
  } catch {
    return value;
  }
}

function renderJobs(jobs) {
  if (!jobsList) return;
  const openDetails = new Set(
    [...jobsList.querySelectorAll("details[open][data-detail-key]")].map((detail) => detail.dataset.detailKey)
  );
  if (!jobs.length) {
    jobsList.innerHTML = `<p class="empty">No Codex jobs yet.</p>`;
    return;
  }
  jobsList.innerHTML = jobs.map((job) => {
    const title = job.type.replaceAll("_", " ");
    const promptText = job.payload && (job.payload.prompt || job.payload.instruction);
    const prompt = promptText ? `<p class="job-prompt">${escapeHtml(promptText).slice(0, 180)}</p>` : "";
    const stats = [
      job.generated_count ? `${job.generated_count} generated` : "",
      job.imported_count ? `${job.imported_count} imported` : "",
      job.return_code !== null && job.return_code !== undefined ? `exit ${job.return_code}` : "",
    ].filter(Boolean).join(" · ");
    const outputPath = job.output_path ? `<p class="job-stats">${escapeHtml(job.output_path)}</p>` : "";
    const commandKey = `${job.id}:command`;
    const outputKey = `${job.id}:output`;
    const commandOpen = openDetails.has(commandKey) ? "open" : "";
    const outputOpen = openDetails.has(outputKey) || job.status === "running" ? "open" : "";
    const command = job.command && job.command.length ? `<details data-detail-key="${commandKey}" ${commandOpen}><summary>Command</summary><pre>${escapeHtml(job.command.join(" "))}</pre></details>` : "";
    const cancellable = ["train_anima_lora", "train_qwen_edit_lora", "setup_musubi_tuner", "setup_sd_scripts"].includes(job.type);
    const cancel = cancellable && job.status === "running"
      ? `<form method="post" action="/datasets/${panel.dataset.datasetSlug}/jobs/${job.id}/cancel"><button type="submit" class="danger">Cancel training</button></form>`
      : "";
    const output = (job.output_tail || []).slice(-20).map(escapeHtml).join("\n");
    return `
      <article class="job-card">
        <div class="job-topline">
          <strong>${escapeHtml(title)}</strong>
          ${statusLabel(job.status)}
        </div>
        <div class="job-meta">
          <span>created ${formatDate(job.created_at)}</span>
          ${job.started_at ? `<span>started ${formatDate(job.started_at)}</span>` : ""}
          ${job.finished_at ? `<span>finished ${formatDate(job.finished_at)}</span>` : ""}
        </div>
        ${prompt}
        ${job.message ? `<p>${escapeHtml(job.message)}</p>` : ""}
        ${job.error ? `<p class="job-error">${escapeHtml(job.error)}</p>` : ""}
        ${stats ? `<p class="job-stats">${stats}</p>` : ""}
        ${outputPath}
        ${command}
        <details data-detail-key="${outputKey}" ${outputOpen}>
          <summary>Job output</summary>
          <pre>${output || "Waiting for job output..."}</pre>
        </details>
        ${cancel}
      </article>
    `;
  }).join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function pollJobs() {
  if (!panel) return;
  const slug = panel.dataset.datasetSlug;
  const response = await fetch(`/datasets/${slug}/jobs`);
  if (!response.ok) return;
  const data = await response.json();
  const jobs = data.jobs || [];
  renderJobs(jobs);
  const shouldReload = jobs.some((job) => {
    const completedWithImports = job.status === "success" && Number(job.imported_count || 0) > 0;
    const completedWithCandidates = job.status === "success" && job.type === "curate_raw_images" && Number(job.generated_count || 0) > 0;
    return (completedWithImports || completedWithCandidates) && !completedJobs.has(job.id);
  });
  if (shouldReload) {
    jobs.forEach((job) => {
      if (job.status === "success") completedJobs.add(job.id);
    });
    localStorage.setItem(completedKey, JSON.stringify([...completedJobs]));
    window.location.reload();
  }
}

initialJobs.forEach((job) => {
  if (job.status === "success") completedJobs.add(job.id);
});
if (completedKey) {
  localStorage.setItem(completedKey, JSON.stringify([...completedJobs]));
}
renderJobs(initialJobs);
setInterval(pollJobs, 2000);
pollJobs();
