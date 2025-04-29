document.getElementById('eligibilityForm').addEventListener('submit', async function (event) {
    event.preventDefault();

    const age = document.getElementById('age').value;
    const gender = document.getElementById('gender').value;
    const state = document.getElementById('state').value;
    const income = document.getElementById('income').value;

    const response = await fetch('/recommend', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ age, gender, state, income })
    });

    const recommendations = await response.json();

    const recommendationsDiv = document.getElementById('recommendations');
    recommendationsDiv.innerHTML = '<h2>Top Recommendations:</h2>';
    
    if (recommendations.length === 0) {
        recommendationsDiv.innerHTML += '<p>No recommendations available.</p>';
    } else {
        recommendations.forEach(function (scheme) {
            recommendationsDiv.innerHTML += `<p>- ${scheme}</p>`;
        });
    }
});
