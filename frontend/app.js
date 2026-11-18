/*
 * API contract expected from POST /api/reconstruct:
 * { trip_id, gps_points: [{lat, lng}], predicted_route: [{lat, lng}],
 *   road_network: [[{lat, lng}]],
 *   predicted_arc_ids: number[], metrics: {top_1_accuracy, macro_f1,
 *   route_continuity, inference_ms, candidate_miss_rate}, decoder_note }
 * Set API_BASE_URL to Laptop 1's address before the two-laptop demonstration.
 */
const API_BASE_URL = "";
const USE_DEMO_DATA = true;
const TRIP_IDS = ["00000000", "00000007", "00000009", "00000040", "00000052", "00000056", "00000065", "00000076", "00000077", "00000091"];

const demoByTrip = Object.fromEntries(TRIP_IDS.map((id, index) => [id, createDemoResult(id, index)]));
let selectedTripId = TRIP_IDS[0];
let map;
let layers;

function createDemoResult(tripId, offset) {
  const center = [12.9716 + offset * 0.00055, 77.5946 + offset * 0.00045];
  const route = [[0, 0], [0.0018, 0.0009], [0.0038, 0.00125], [0.0057, 0.0028], [0.0077, 0.004], [0.0096, 0.0044]].map(([lat, lng]) => ({ lat: center[0] + lat, lng: center[1] + lng }));
  const gps = route.map((point, i) => ({ lat: point.lat + [0.00013, -0.00009, 0.00019, -0.00015, 0.00008, -0.00011][i], lng: point.lng + [-0.00011, 0.00014, -0.0001, 0.00013, -0.00008, 0.00012][i] }));
  return { trip_id: tripId, gps_points: gps, predicted_route: route, road_network: createDemoNetwork(route), predicted_arc_ids: [142 + offset, 143 + offset, 158 + offset, 159 + offset, 160 + offset], metrics: { top_1_accuracy: 0.84 + (offset % 4) * 0.02, macro_f1: 0.78 + (offset % 3) * 0.025, route_continuity: 0.92 + (offset % 3) * 0.02, inference_ms: 128 + offset * 7, candidate_miss_rate: 0.04 + (offset % 3) * 0.01 }, decoder_note: "Viterbi selected the highest-probability connected arc sequence from the classifier candidates." };
}

function createDemoNetwork(route) {
  const origin = route[0];
  const horizontal = [-0.0015, 0.001, 0.0034, 0.006, 0.0085].map(offset => [
    { lat: origin.lat + offset, lng: origin.lng - 0.002 },
    { lat: origin.lat + offset, lng: origin.lng + 0.007 }
  ]);
  const vertical = [-0.001, 0.0015, 0.004, 0.0062].map(offset => [
    { lat: origin.lat - 0.002, lng: origin.lng + offset },
    { lat: origin.lat + 0.0105, lng: origin.lng + offset }
  ]);
  return [...horizontal, ...vertical];
}

function initMap() {
  map = L.map("map", { zoomControl: false, attributionControl: false }).setView([12.9716, 77.5946], 14);
  L.control.zoom({ position: "bottomright" }).addTo(map);
  layers = L.layerGroup().addTo(map);
}

function renderTripButtons() {
  document.getElementById("tripList").innerHTML = TRIP_IDS.map(id => `<button class="trip-button ${id === selectedTripId ? "is-selected" : ""}" type="button" data-trip-id="${id}">${id.slice(-2)}</button>`).join("");
  document.querySelectorAll(".trip-button").forEach(button => button.addEventListener("click", () => {
    selectedTripId = button.dataset.tripId;
    document.getElementById("requestNote").textContent = `Ready to run trip ${selectedTripId}.`;
    document.getElementById("mapTripTitle").textContent = `Trip ${selectedTripId}`;
    renderTripButtons();
  }));
}

function renderResult(result) {
  const routeSegments = result.predicted_route_segments || [result.predicted_route || []];
  const route = routeSegments.flat().map(point => [point.lat, point.lng]);
  const gps = result.gps_points.map(point => [point.lat, point.lng]);
  layers.clearLayers();
  (result.road_network || []).forEach(segment => {
    L.polyline(segment.map(point => [point.lat, point.lng]), { color: "#bdcbd8", weight: 3, opacity: .85 }).addTo(layers);
  });
  routeSegments.filter(segment => segment.length > 1).forEach(segment => {
    L.polyline(segment.map(point => [point.lat, point.lng]), { color: "#1f7aee", weight: 6, opacity: .9 }).addTo(layers);
  });
  L.polyline(gps, { color: "#f6a623", weight: 2, opacity: .85, dashArray: "4 7" }).addTo(layers);
  gps.forEach(point => L.circleMarker(point, { radius: 4, color: "#ffffff", weight: 1, fillColor: "#f6a623", fillOpacity: 1 }).addTo(layers));
  if (route.length) {
    L.circleMarker(route[0], { radius: 8, color: "#ffffff", weight: 3, fillColor: "#0fa3b1", fillOpacity: 1 }).bindTooltip("Start").addTo(layers);
    L.circleMarker(route.at(-1), { radius: 8, color: "#ffffff", weight: 3, fillColor: "#102a43", fillOpacity: 1 }).bindTooltip("Finish").addTo(layers);
  }
  map.fitBounds(L.latLngBounds([...route, ...gps]), { padding: [40, 40], maxZoom: 16 });

  const m = result.metrics;
  const cards = [
    ["Top-1 accuracy", percent(m.top_1_accuracy), "Candidate classification"],
    ["Macro F1", percent(m.macro_f1), "Across observed points"],
    ["Route continuity", percent(m.route_continuity), "Connected transitions"],
    ["Candidate miss rate", percent(m.candidate_miss_rate), "Reference not proposed"],
    ["Inference time", `${m.inference_ms} ms`, "End-to-end decoding"]
  ];
  document.getElementById("metrics").innerHTML = cards.map(([label, value, detail]) => `<article class="metric-card"><p class="metric-label">${label}</p><p class="metric-value">${value}</p><p class="metric-detail">${detail}</p></article>`).join("");
  document.getElementById("mapTripTitle").textContent = `Trip ${result.trip_id}`;
  document.getElementById("mapCaption").textContent = `${result.gps_points.length} GPS observations decoded into ${result.predicted_arc_ids.length} connected road arcs.`;
  document.getElementById("routeStatus").textContent = "Route decoded";
  document.getElementById("routeStatus").classList.add("is-ready");
  document.getElementById("updatedAt").textContent = `Last run: ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
  document.getElementById("decoderNote").textContent = result.decoder_note;
  document.getElementById("arcSequence").textContent = result.predicted_arc_ids.join("  →  ");
}

function percent(value) { return `${(value * 100).toFixed(1)}%`; }

async function reconstructSelectedTrip() {
  const button = document.getElementById("runButton");
  button.disabled = true;
  document.getElementById("requestNote").textContent = `Running route reconstruction for ${selectedTripId}…`;
  document.getElementById("routeStatus").textContent = "Processing";
  try {
    let result;
    if (USE_DEMO_DATA) {
      await new Promise(resolve => setTimeout(resolve, 550));
      result = demoByTrip[selectedTripId];
    } else {
      const response = await fetch(`${API_BASE_URL}/api/reconstruct`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trip_id: selectedTripId }) });
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      result = await response.json();
    }
    renderResult(result);
    document.getElementById("requestNote").textContent = `Completed reconstruction for ${selectedTripId}.`;
  } catch (error) {
    document.getElementById("routeStatus").textContent = "Request failed";
    document.getElementById("requestNote").textContent = `Could not reconstruct ${selectedTripId}: ${error.message}`;
  } finally { button.disabled = false; }
}

function init() {
  initMap();
  renderTripButtons();
  document.getElementById("runButton").addEventListener("click", reconstructSelectedTrip);
  document.getElementById("connectionStatus").lastElementChild.textContent = USE_DEMO_DATA ? "Demo data" : "API connected";
}

init();
