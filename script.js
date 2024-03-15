document.getElementById("startButton").addEventListener("click", () => {
    document.getElementById("startButton").style.display = "none";
    document.getElementById("question").innerText = "What's your answer?";
    document.getElementById("uploadButton").style.display = "block";
    document.getElementById("recordButton").style.display = "block";
});

document.getElementById("uploadButton").addEventListener("click", () => {
    document.getElementById("fileInput").click();
});

document.getElementById("fileInput").addEventListener("change", (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append("audio", file);
    
    fetch("/process_audio", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("transcription").innerText = data.transcription;
        document.getElementById("sentiment").innerText = JSON.stringify(data.sentiment, null, 2);
        document.getElementById("results").style.display = "block";
    })
    .catch(error => console.error("Error:", error));
});

document.getElementById("recordButton").addEventListener("click", () => {
    fetch("/process_audio", {
        method: "POST"
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("transcription").innerText = data.transcription;
        document.getElementById("sentiment").innerText = JSON.stringify(data.sentiment, null, 2);
        document.getElementById("results").style.display = "block";
    })
    .catch(error => console.error("Error:", error));
});
