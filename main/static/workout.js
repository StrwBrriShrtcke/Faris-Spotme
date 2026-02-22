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

console.log("Loaded:", pageTitle, video);
