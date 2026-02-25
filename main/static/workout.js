console.log("loading workout");

// Page Init
console.log(window.location.search);
const params = new URLSearchParams(window.location.search);
const pageTitle = params.get("pageTitle") || "error";
const video = params.get("video") || "error";

if (pageTitle === "error") {
  alert("Invalid workout selection");
  window.location.href = "/selection";
}

document.title = pageTitle;
document.getElementById("bodyTitle").innerText = pageTitle;
document.getElementById("workout-video").src = video;

const socket = io();

socket.on("connect", () => {
  socket.emit("set_exercise", { exercise: pageTitle.toLowerCase() });
});

socket.on("pose_data", (data) => {
  const leftAngle =
    data.left_angle === null || data.left_angle === undefined
      ? "—"
      : Number(data.left_angle).toFixed(1);

  const rightAngle =
    data.right_angle === null || data.right_angle === undefined
      ? "—"
      : Number(data.right_angle).toFixed(1);

  document.getElementById("left-angle").textContent = leftAngle;
  document.getElementById("right-angle").textContent = rightAngle;

  document.getElementById("left-feedback").textContent =
    data.left_feedback ?? "—";
  document.getElementById("right-feedback").textContent =
    data.right_feedback ?? "—";
});

const img = document.getElementById("pose-camera");

img.addEventListener("click", (e) => {
  const rect = img.getBoundingClientRect();

  // click location within the displayed image
  const xDisp = e.clientX - rect.left;
  const yDisp = e.clientY - rect.top;

  // scale to the server frame size
  // IMPORTANT: set these to whatever cap is producing (480x360 in your app)
  const FRAME_W = 480; // <-- change to match your cap width
  const FRAME_H = 360; // <-- change to match your cap height

  const x = Math.round((xDisp / rect.width) * FRAME_W);
  const y = Math.round((yDisp / rect.height) * FRAME_H);

  socket.emit("select_point", { x, y });
});

console.log("Loaded:", pageTitle, video);
