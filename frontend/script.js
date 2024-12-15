document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const imageUpload = document.getElementById('image-upload');
    const predictionsContainer = document.getElementById('predictions');
    const loadingIndicator = document.getElementById('loading');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Reset previous results
        predictionsContainer.innerHTML = '';
        loadingIndicator.classList.remove('hidden');

        // Prepare form data
        const formData = new FormData();
        formData.append('file', imageUpload.files[0]);

        try {
            // Send request to backend
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            // Hide loading
            loadingIndicator.classList.add('hidden');

            // Display predictions
            if (data.predictions) {
                data.predictions.forEach((pred, index) => {
                    const predictionElement = document.createElement('div');
                    predictionElement.classList.add('prediction-item', 'p-4', 'bg-blue-50', 'rounded-lg');
                    predictionElement.innerHTML = `
                        <p class="font-bold">${pred.diagnosis}</p>
                        <p>Probability: ${(pred.probability * 100).toFixed(2)}%</p>
                    `;
                    predictionsContainer.appendChild(predictionElement);
                });
            }
        } catch (error) {
            // Hide loading and show error
            loadingIndicator.classList.add('hidden');
            predictionsContainer.innerHTML = `
                <div class="text-red-500">
                    Error: ${error.message}
                </div>
            `;
        }
    });
});