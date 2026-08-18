async function predictEmotion() {

    const text =
        document.getElementById("text").value;


    if (text.trim() === "") {

        document.getElementById("result").innerHTML =
            "Please enter some text.";

        return;

    }


    const response = await fetch(
        "/predict",
        {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: text
            })

        }
    );


    const data =
        await response.json();


    let emoji = "😐";


    if (data.emotion === "happy")
        emoji = "😊";

    else if (data.emotion === "sad")
        emoji = "😢";

    else if (data.emotion === "angry")
        emoji = "😡";

    else if (data.emotion === "fear")
        emoji = "😨";

    else if (data.emotion === "love")
        emoji = "❤️";

    else if (data.emotion === "surprise")
        emoji = "😲";


    document.getElementById("result").innerHTML =

        emoji +
        " " +
        data.emotion.toUpperCase() +
        "<br><br>" +

        "Confidence: " +
        data.confidence +
        "%";

}