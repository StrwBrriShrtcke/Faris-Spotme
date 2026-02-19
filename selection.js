console.log("hello");

// Init Element Variables
const workoutTypes = {
  bicepCurl: {
    element: document.getElementById("bicep-curl"),
    name: "Bicep Curl",
    video: "https://www.youtube.com/embed/rZM88p-VZe8",
  },
  rows: {
    element: document.getElementById("rows"),
    name: "Rows",
    video: "https://www.youtube.com/embed/xQNrFHEMhI4",
  },
  abbCrunch: {
    element: document.getElementById("abb-crunch"),
    name: "Abb Crunch",
    video: "https://www.youtube.com/embed/m_TnNDJAUF8",
  },
  legCurl: {
    element: document.getElementById("leg-curl"),
    name: "Leg Curl",
    video: "https://www.youtube.com/embed/t9sTSr-JYSs",
  },
  chestPress: {
    element: document.getElementById("chest-press"),
    name: "Chest Press",
    video: "https://www.youtube.com/embed/NwzUje3z0qY",
  },
  TricepPress: {
    element: document.getElementById("tricep-press"),
    name: "Tricep Press",
    video: "https://www.youtube.com/embed/yPohcZXCZSc",
  },
};

// Event Listener
let createEventListener = (workoutType) => {
  console.log(`${workoutType.name} add event listener`);
  workoutType.element.addEventListener("click", () => {
    window.location.href = `workout.html?pageTitle=${workoutType.name + "&video=" + workoutType.video}`;
    const video = workoutType.video;
    console.log(video, "video");
  });
};

console.log("----------------");
for (let workout of Object.values(workoutTypes)) {
  createEventListener(workout);
}
