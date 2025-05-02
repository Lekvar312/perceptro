let model;
let chart;

function createChart() {
    const ctx = document.getElementById('lossChart').getContext('2d');
    if (chart) chart.destroy();
    chart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Втрата', data: [], borderColor: 'red', fill: false }] },
        options: { responsive: true, scales: { x: { title: { text: 'Епоха', display: true } }, y: { title: { text: 'Втрата', display: true } } } }
    });
    return chart;
}

function updateChart(chart, losses) {
    chart.data.labels = losses.map((_, i) => i + 1);
    chart.data.datasets[0].data = losses;
    chart.update();
}

async function trainModel() {
    document.getElementById('status').innerText = "Статус: Навчання...";

    const trainXs = tf.tensor2d([
        [150, 1, 0, 0, 0.7], // Яблуко
        [120, 1, 0.2, 0.1, 0.8], // Яблуко
        [200, 1, 1, 0, 0.9], // Банан
        [180, 1, 1, 0.2, 0.85], // Банан
        [50, 0.5, 0, 0.5, 0.6], // Виноград
        [40, 0.4, 0.1, 0.7, 0.5], // Виноград
    ]);
    
    const trainYs = tf.tensor2d([
        [1, 0, 0], // Яблуко
        [1, 0, 0], // Яблуко
        [0, 1, 0], // Банан
        [0, 1, 0], // Банан
        [0, 0, 1], // Виноград
        [0, 0, 1], // Виноград
    ]);

    model = tf.sequential();
    model.add(tf.layers.dense({ units: 8, activation: 'relu', inputShape: [5] }));
    model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    
    model.compile({ optimizer: tf.train.adam(0.01), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    const losses = [];
    const chart = createChart();

    await model.fit(trainXs, trainYs, {
        epochs: 150,
        batchSize: 4,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(4)}`);
                losses.push(logs.loss);
                updateChart(chart, losses);
            }
        }
    });
    
    document.getElementById('status').innerText = "Статус: Навчання завершено!";
}

function predict() {
    const inputData = [
        parseFloat(document.getElementById('input1').value),
        parseFloat(document.getElementById('input2').value),
        parseFloat(document.getElementById('input3').value),
        parseFloat(document.getElementById('input4').value),
        parseFloat(document.getElementById('input5').value)
    ];

    let isValid = true;
    const inputs = [document.getElementById('input1'), document.getElementById('input2'), document.getElementById('input3'), document.getElementById('input4'), document.getElementById('input5')];

    inputs.forEach(input => input.style.borderColor = "");

    if (inputData.some(isNaN) || inputData[0] <= 0 || inputData.slice(1).some(val => val < 0 || val > 1)) {
        document.getElementById('prediction').innerText = "Помилка: некоректні вхідні дані";

        inputs.forEach((input, index) => {
            if (isNaN(inputData[index]) || 
                (index === 0 && inputData[index] <= 0) || 
                (index > 0 && (inputData[index] < 0 || inputData[index] > 1))) {
                input.style.borderColor = "red";
            }
        });
        return;
    }

    const inputTensor = tf.tensor2d([inputData]);
    const prediction = model.predict(inputTensor);

    const predictedClass = prediction.argMax(1).dataSync()[0];
    const labels = ['Яблуко', 'Банан', 'Виноград'];
    document.getElementById('prediction').innerText = `Модель передбачила: ${labels[predictedClass]}`;
}


function openModal() {
    document.getElementById("modal").style.display = "block";
}

function closeModal() {
    document.getElementById("modal").style.display = "none";
}

// Закриття при кліку поза вікном
window.onclick = function(event) {
    const modal = document.getElementById("modal");
    if (event.target === modal) {
        modal.style.display = "none";
    }
};
