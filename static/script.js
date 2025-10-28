document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("form");
    const predictionText = document.querySelector("h3");

    form.addEventListener("submit", () => {
        predictionText.innerHTML = "🔎 Analyzing data... please wait...";
    });
});
