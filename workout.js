console.log("working out");

// Page Init
console.log(window.location.search);
const pageTitle =
  new URLSearchParams(window.location.search).get("pageTitle") || "error";
const video =
  new URLSearchParams(window.location.search).get("video") || "error";

if (pageTitle === "error") {
  alert("Ah fuck");
  window.location.href = "selection.html";
}

document.getElementById("pageTitle").innerText = pageTitle;
document.getElementById("bodyTitle").innerText = pageTitle;
console.log(video, "video");
document.getElementById("workout-video").src = video;
