const DATA_URL = "./data/benchmark-data.json";
const VALID_VERSIONS = new Set(["ver1", "ver2"]);
const UI_FONT_FAMILY = "\"IBM Plex Sans\", \"Segoe UI\", sans-serif";
const NUMERIC_FONT_FAMILY = "\"JetBrains Mono\", \"IBM Plex Sans\", \"Segoe UI\", sans-serif";

const state = {
  metadata: {},
  entries: [],
  filtered: [],
  filters: {
    version: "ver2",
    vendor: "all",
    architecture: "all",
    platform: "all",
    search: "",
    sort: "fp32-desc",
  },
  charts: {
    topDevices: null,
    vendor: null,
  },
};

function inferVersionFromNote(note) {
  return (note || "").toLowerCase().includes("ver.2") ? "ver2" : "ver1";
}

function configureChartDefaults() {
  if (typeof Chart === "undefined") {
    return;
  }
  Chart.defaults.font.family = UI_FONT_FAMILY;
  Chart.defaults.color = "#334155";
}

function parseDateKey(value) {
  const parts = String(value || "").split(".").map((part) => Number(part));
  if (parts.length !== 3 || parts.some((part) => Number.isNaN(part))) {
    return 0;
  }
  const [year, month, day] = parts;
  return year * 10000 + month * 100 + day;
}

function formatScore(value) {
  return typeof value === "number" ? value.toLocaleString() : "N/A";
}

function formatBatchSize(value) {
  if (value === 0) {
    return "Auto";
  }
  if (typeof value === "number") {
    return value.toLocaleString();
  }
  return "N/A";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function toTitleCase(text) {
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

function setMetric(id, value) {
  const el = document.getElementById(id);
  if (!el) {
    return;
  }
  el.textContent = value;
  el.classList.remove("skeleton");
}

function normalizeEntry(entry) {
  const version = VALID_VERSIONS.has(entry.version) ? entry.version : inferVersionFromNote(entry.note);
  return {
    ...entry,
    version,
  };
}

function populateSelect(id, values, formatter = (value) => value) {
  const select = document.getElementById(id);
  if (!select) {
    return;
  }

  const currentValue = select.value;
  const firstOption = select.options[0];
  select.innerHTML = "";
  if (firstOption) {
    select.appendChild(firstOption);
  }

  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = formatter(value);
    select.appendChild(option);
  });

  if (Array.from(select.options).some((option) => option.value === currentValue)) {
    select.value = currentValue;
  }
}

function getArchitectureScopedEntries() {
  return state.entries.filter((entry) => {
    if (state.filters.version !== "all" && entry.version !== state.filters.version) {
      return false;
    }
    if (state.filters.vendor !== "all" && entry.vendor !== state.filters.vendor) {
      return false;
    }
    return true;
  });
}

function refreshArchitectureOptions() {
  const architectures = [...new Set(getArchitectureScopedEntries().map((entry) => entry.architecture))].sort();
  populateSelect("architectureFilter", architectures);

  if (state.filters.architecture !== "all" && !architectures.includes(state.filters.architecture)) {
    state.filters.architecture = "all";
  }

  const architectureFilter = document.getElementById("architectureFilter");
  if (architectureFilter) {
    architectureFilter.value = state.filters.architecture;
  }
}

function updateVersionPills() {
  const pills = Array.from(document.querySelectorAll(".version-pill"));
  pills.forEach((pill) => {
    const isActive = pill.dataset.version === state.filters.version;
    pill.classList.toggle("active", isActive);
    pill.setAttribute("aria-selected", String(isActive));
    pill.tabIndex = isActive ? 0 : -1;
  });
}

function bindEvents() {
  const vendorFilter = document.getElementById("vendorFilter");
  const architectureFilter = document.getElementById("architectureFilter");
  const platformFilter = document.getElementById("platformFilter");
  const sortBy = document.getElementById("sortBy");
  const searchInput = document.getElementById("searchInput");
  const versionSwitch = document.getElementById("versionSwitch");

  vendorFilter?.addEventListener("change", (event) => {
    state.filters.vendor = event.target.value;
    refreshArchitectureOptions();
    applyFilters();
  });

  architectureFilter?.addEventListener("change", (event) => {
    state.filters.architecture = event.target.value;
    applyFilters();
  });

  platformFilter?.addEventListener("change", (event) => {
    state.filters.platform = event.target.value;
    applyFilters();
  });

  sortBy?.addEventListener("change", (event) => {
    state.filters.sort = event.target.value;
    applyFilters();
  });

  searchInput?.addEventListener("input", (event) => {
    state.filters.search = event.target.value.trim().toLowerCase();
    applyFilters();
  });

  versionSwitch?.addEventListener("click", (event) => {
    const target = event.target.closest(".version-pill");
    if (!target) {
      return;
    }
    state.filters.version = target.dataset.version;
    updateVersionPills();
    refreshArchitectureOptions();
    applyFilters();
  });

  versionSwitch?.addEventListener("keydown", (event) => {
    const pills = Array.from(versionSwitch.querySelectorAll(".version-pill"));
    const currentIndex = pills.indexOf(document.activeElement);
    if (currentIndex < 0) {
      return;
    }

    let nextIndex = currentIndex;
    if (event.key === "ArrowRight") {
      nextIndex = (currentIndex + 1) % pills.length;
    } else if (event.key === "ArrowLeft") {
      nextIndex = (currentIndex - 1 + pills.length) % pills.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = pills.length - 1;
    } else if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      document.activeElement.click();
      return;
    } else {
      return;
    }

    event.preventDefault();
    pills[nextIndex].focus();
  });
}

function sortFilteredData() {
  const sortBy = state.filters.sort;

  const getNumber = (value, fallback) => (typeof value === "number" ? value : fallback);

  if (sortBy === "fp32-desc") {
    state.filtered.sort((a, b) => getNumber(b.fp32, Number.NEGATIVE_INFINITY) - getNumber(a.fp32, Number.NEGATIVE_INFINITY));
  } else if (sortBy === "fp32-asc") {
    state.filtered.sort((a, b) => getNumber(a.fp32, Number.POSITIVE_INFINITY) - getNumber(b.fp32, Number.POSITIVE_INFINITY));
  } else if (sortBy === "fp16-desc") {
    state.filtered.sort((a, b) => getNumber(b.fp16, Number.NEGATIVE_INFINITY) - getNumber(a.fp16, Number.NEGATIVE_INFINITY));
  } else if (sortBy === "fp16-asc") {
    state.filtered.sort((a, b) => getNumber(a.fp16, Number.POSITIVE_INFINITY) - getNumber(b.fp16, Number.POSITIVE_INFINITY));
  } else if (sortBy === "date-desc") {
    state.filtered.sort((a, b) => parseDateKey(b.date) - parseDateKey(a.date));
  } else if (sortBy === "date-asc") {
    state.filtered.sort((a, b) => parseDateKey(a.date) - parseDateKey(b.date));
  } else if (sortBy === "device") {
    state.filtered.sort((a, b) => a.device.localeCompare(b.device));
  }
}

function applyFilters() {
  const { version, vendor, architecture, platform, search } = state.filters;

  state.filtered = state.entries.filter((entry) => {
    if (version !== "all" && entry.version !== version) {
      return false;
    }
    if (vendor !== "all" && entry.vendor !== vendor) {
      return false;
    }
    if (architecture !== "all" && entry.architecture !== architecture) {
      return false;
    }
    if (platform !== "all" && entry.platform !== platform) {
      return false;
    }

    if (!search) {
      return true;
    }

    const searchFields = [
      entry.device,
      entry.platform,
      entry.note,
      entry.architecture,
      entry.version,
      entry.vendor,
      entry.memory,
    ]
      .join(" ")
      .toLowerCase();

    return searchFields.includes(search);
  });

  sortFilteredData();
  renderStats();
  renderTable();
  renderCharts();
}

function renderMetadata() {
  const metadata = state.metadata || {};
  const testConfig = metadata.testConfiguration || {};

  const updated = metadata.lastUpdated || "Unknown";
  const lastUpdatedEl = document.getElementById("lastUpdated");
  if (lastUpdatedEl) {
    lastUpdatedEl.textContent = `Last updated: ${updated}`;
  }

  const modelName = document.getElementById("modelName");
  if (modelName) {
    modelName.textContent = testConfig.model || "ResNet50";
  }

  const parameterCount = document.getElementById("parameterCount");
  if (parameterCount) {
    parameterCount.textContent = testConfig.parameters || "23.5M";
  }

  const defaultCommand = document.getElementById("defaultCommand");
  if (defaultCommand) {
    defaultCommand.textContent = testConfig.defaultCommand || "make ddp-abs";
  }
}

function renderStats() {
  const fp32 = state.filtered.filter((entry) => typeof entry.fp32 === "number").map((entry) => entry.fp32);
  const fp16 = state.filtered.filter((entry) => typeof entry.fp16 === "number").map((entry) => entry.fp16);
  const vendors = new Set(state.filtered.map((entry) => entry.vendor));

  setMetric("visibleEntries", state.filtered.length.toLocaleString());
  setMetric("topFP32", fp32.length ? Math.max(...fp32).toLocaleString() : "N/A");
  setMetric("topFP16", fp16.length ? Math.max(...fp16).toLocaleString() : "N/A");
  setMetric("vendorCount", vendors.size.toLocaleString());
}

function renderTable() {
  const tableBody = document.getElementById("tableBody");
  const emptyState = document.getElementById("emptyState");
  const resultCount = document.getElementById("resultCount");

  if (!tableBody) {
    return;
  }

  resultCount.textContent = `${state.filtered.length.toLocaleString()} entries`;

  if (state.filtered.length === 0) {
    tableBody.innerHTML = "";
    emptyState.hidden = false;
    return;
  }

  emptyState.hidden = true;

  tableBody.innerHTML = state.filtered
    .map(
      (entry) => `
      <tr>
        <td>
          <div class="device-name">${escapeHtml(entry.device)}</div>
          ${entry.memory ? `<small>${escapeHtml(entry.memory)}</small>` : ""}
        </td>
        <td><span class="version-badge ${escapeHtml(entry.version)}">${escapeHtml(entry.version.toUpperCase())}</span></td>
        <td>${escapeHtml(entry.platform || "N/A")}</td>
        <td class="metric fp32">${formatScore(entry.fp32)}</td>
        <td><span class="bs-tag">${escapeHtml(formatBatchSize(entry.fp32bs))}</span></td>
        <td class="metric fp16">${formatScore(entry.fp16)}</td>
        <td><span class="bs-tag">${escapeHtml(formatBatchSize(entry.fp16bs))}</span></td>
        <td class="note">${escapeHtml(entry.note || "N/A")}</td>
        <td>${escapeHtml(entry.date || "N/A")}</td>
      </tr>
    `,
    )
    .join("");
}

function renderTopDevicesChart() {
  if (typeof Chart === "undefined") {
    return;
  }

  const canvas = document.getElementById("topDevicesChart");
  if (!canvas) {
    return;
  }

  const topEntries = [...state.filtered]
    .filter((entry) => typeof entry.fp32 === "number")
    .sort((a, b) => b.fp32 - a.fp32)
    .slice(0, 12);

  const labels = topEntries.map((entry) => {
    const label = entry.device;
    return label.length > 26 ? `${label.slice(0, 26)}…` : label;
  });

  if (state.charts.topDevices) {
    state.charts.topDevices.destroy();
  }

  state.charts.topDevices = new Chart(canvas.getContext("2d"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "FP32",
          data: topEntries.map((entry) => entry.fp32),
          backgroundColor: "rgba(15, 118, 110, 0.75)",
          borderColor: "rgba(15, 118, 110, 1)",
          borderWidth: 1,
        },
        {
          label: "FP16",
          data: topEntries.map((entry) => (typeof entry.fp16 === "number" ? entry.fp16 : 0)),
          backgroundColor: "rgba(3, 105, 161, 0.6)",
          borderColor: "rgba(3, 105, 161, 1)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: {
            maxRotation: 35,
            minRotation: 35,
          },
        },
        y: {
          beginAtZero: true,
          ticks: {
            font: {
              family: NUMERIC_FONT_FAMILY,
            },
          },
          title: {
            display: true,
            text: "Score",
          },
        },
      },
      plugins: {
        legend: {
          position: "top",
        },
        tooltip: {
          callbacks: {
            title(items) {
              const index = items[0].dataIndex;
              return topEntries[index].device;
            },
          },
        },
      },
    },
  });
}

function renderVendorChart() {
  if (typeof Chart === "undefined") {
    return;
  }

  const canvas = document.getElementById("vendorChart");
  if (!canvas) {
    return;
  }

  const aggregates = new Map();
  state.filtered.forEach((entry) => {
    if (!aggregates.has(entry.vendor)) {
      aggregates.set(entry.vendor, {
        count: 0,
        fp32Sum: 0,
        fp32Count: 0,
        fp32Peak: 0,
      });
    }

    const current = aggregates.get(entry.vendor);
    current.count += 1;
    if (typeof entry.fp32 === "number") {
      current.fp32Sum += entry.fp32;
      current.fp32Count += 1;
      current.fp32Peak = Math.max(current.fp32Peak, entry.fp32);
    }
  });

  const sorted = [...aggregates.entries()].sort(([, a], [, b]) => b.fp32Peak - a.fp32Peak);
  const labels = sorted.map(([vendor]) => toTitleCase(vendor));
  const peaks = sorted.map(([, stats]) => stats.fp32Peak);
  const averages = sorted.map(([, stats]) =>
    stats.fp32Count > 0 ? Math.round(stats.fp32Sum / stats.fp32Count) : 0,
  );

  if (state.charts.vendor) {
    state.charts.vendor.destroy();
  }

  state.charts.vendor = new Chart(canvas.getContext("2d"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Peak FP32",
          data: peaks,
          backgroundColor: "rgba(15, 118, 110, 0.72)",
          borderColor: "rgba(15, 118, 110, 1)",
          borderWidth: 1,
        },
        {
          label: "Average FP32",
          data: averages,
          backgroundColor: "rgba(245, 158, 11, 0.55)",
          borderColor: "rgba(245, 158, 11, 1)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            font: {
              family: NUMERIC_FONT_FAMILY,
            },
          },
          title: {
            display: true,
            text: "Score",
          },
        },
      },
      plugins: {
        legend: {
          position: "top",
        },
      },
    },
  });
}

function renderCharts() {
  renderTopDevicesChart();
  renderVendorChart();
}

function showError(message) {
  const errorState = document.getElementById("errorState");
  const errorText = document.getElementById("errorText");
  if (errorText) {
    errorText.textContent = message;
  }
  if (errorState) {
    errorState.hidden = false;
  }
}

function initFilterOptions() {
  const vendors = [...new Set(state.entries.map((entry) => entry.vendor))].sort();
  const platforms = [...new Set(state.entries.map((entry) => entry.platform))].sort();

  populateSelect("vendorFilter", vendors, toTitleCase);
  populateSelect("platformFilter", platforms);
  refreshArchitectureOptions();
}

async function loadData() {
  try {
    const response = await fetch(DATA_URL, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} while loading benchmark data`);
    }

    const payload = await response.json();
    state.metadata = payload.metadata || {};
    state.entries = Array.isArray(payload.benchmarks) ? payload.benchmarks.map(normalizeEntry) : [];

    configureChartDefaults();
    renderMetadata();
    initFilterOptions();
    updateVersionPills();
    applyFilters();
  } catch (error) {
    showError(`Could not load benchmark data: ${error.message}`);
    setMetric("visibleEntries", "N/A");
    setMetric("topFP32", "N/A");
    setMetric("topFP16", "N/A");
    setMetric("vendorCount", "N/A");
    console.error(error);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  loadData();
});
