document.getElementById("image").addEventListener("change", function (event) {
  const file = event.target.files[0];
  const preview = document.getElementById("preview");

  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      preview.src = e.target.result;
      preview.style.display = "block";
    };
    reader.readAsDataURL(file);
  } else {
    preview.src = "";
    preview.style.display = "none";
  }
});

async function getAnswer() {
  const question = document.getElementById("question").value;
  const imageInput = document.getElementById("image").files[0];
  const responseDiv = document.getElementById("response");

  responseDiv.innerHTML = "Reading input...";

  let imageBase64 = null;
  if (imageInput) {
    try {
      imageBase64 = await readFileAsBase64(imageInput);
    } catch (err) {
      responseDiv.innerHTML = `<p style="color:red;">Error reading image: ${err.message}</p>`;
      return;
    }
  }

  const body = { question };
  if (imageBase64) body.image = imageBase64;

  responseDiv.innerHTML = "<p><i>Tadashi is thinking...</i></p>";

  try {
    const res = await fetch("http://127.0.0.1:8000/api/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

    const data = await res.json();

    if (data.error) {
      responseDiv.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
    } else {
      const linksHTML = (data.links || []).map(link =>
        `<li><a href="${link.url}" target="_blank">${link.text}</a></li>`
      ).join("");

      responseDiv.innerHTML = `
        <p><b>${data.answer}</b></p>
        ${linksHTML ? `<ul>${linksHTML}</ul>` : ""}
      `;
    }
  } catch (error) {
    responseDiv.innerHTML = `<p style="color:red;">Request failed: ${error.message}</p>`;
  }
}

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
