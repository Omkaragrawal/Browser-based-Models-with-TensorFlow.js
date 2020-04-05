import {
	MnistData
} from './data.js';
let canvas, ctx, saveButton, clearButton;
let pos = {
	x: 0,
	y: 0
};
let rawImage;
let model;


const getModel = () => {
	model = tf.sequential();

	model.add(tf.layers.conv2d({
		inputShape: [28, 28, 1],
		kernelSize: 3,
		filters: 8,
		activation: 'relu'
	}));
	model.add(tf.layers.maxPooling2d({
		poolSize: [2, 2]
	}));

	model.add(tf.layers.conv2d({
		filters: 16,
		kernelSize: 3,
		activation: 'relu'
	}));
	model.add(tf.layers.maxPooling2d({
		poolSize: [2, 2]
	}));

	model.add(tf.layers.flatten());

	model.add(tf.layers.dense({
		units: 128,
		activation: 'relu'
	}));
	
	model.add(tf.layers.dense({
		units: 10,
		activation: 'softmax'
	}));

	model.compile({
		optimizer: tf.train.adam(),
		loss: 'categoricalCrossentropy',
		metrics: ['accuracy']
	});

	return model;
}

const train = async (model, data) => {
	const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
	const container = {
		name: 'Model Training',
		styles: {
			height: '700px',
			width: '700px'
		}
	}
	const fitCallbacks = tfvis.show.fitCallbacks(container, metrics)

	const BATCH_SIZE = 512;
	const TRAIN_DATA_SIZE = 5500;
	const TEST_DATA_SIZE = 1000;

	const [trainXs, trainYs] = tf.tidy(() => {
		const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
		return [
			d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
			d.labels
		]
	});

	const [testXs, testYs] = tf.tidy(() => {
		const d = data.nextTestBatch(TEST_DATA_SIZE);
		return [
			d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
			d.labels
		];
	});

	return model.fit(trainXs, trainYs, {
		batchSize: BATCH_SIZE,
		validationData: [testXs, testYs],
		epochs: 20,
		shuffle: true,
		callbacks: fitCallbacks
	});
}

const setPosition = (e) => {
	pos.x = e.clientX - 100;
	pos.y = e.clientY - 100;
}

const draw = (e) => {
	if (e.buttons != 1) return;
	ctx.beginPath();
	ctx.lineWidth = 24;
	ctx.lineCap = 'round';
	ctx.strokeStyle = 'white';
	ctx.moveTo(pos.x, pos.y);
	setPosition(e);
	ctx.lineTo(pos.x, pos.y);
	ctx.stroke();
	rawImage.src = canvas.toDataURL('image/png');
}

const erase = () => {
	ctx.fillStyle = "black";
	ctx.fillRect(0, 0, 280, 280);
}

const save = () => {
	let raw = tf.browser.fromPixels(rawImage, 1);
	let resized = tf.image.resizeBilinear(raw, [28, 28]);
	let tensor = resized.expandDims(0);
	let prediction = model.predict(tensor);
	let pIndex = tf.argMax(prediction, 1).dataSync();

	alert(pIndex);
}

const init = () => {
	canvas = document.getElementById('canvas');
	rawImage = document.getElementById('canvasimg');
	ctx = canvas.getContext("2d");
	ctx.fillStyle = "black";
	ctx.fillRect(0, 0, 280, 280);
	canvas.addEventListener("mousemove", draw);
	canvas.addEventListener("mousedown", setPosition);
	canvas.addEventListener("mouseenter", setPosition);
	saveButton = document.getElementById('sb');
	saveButton.addEventListener("click", save);
	clearButton = document.getElementById('cb');
	clearButton.addEventListener("click", erase);
}

const run = async () => {
	const data = new MnistData();
	await data.load();
	const model = getModel();
	tfvis.show.modelSummary({
		name: 'Model Architecture'
	}, model);
	await train(model, data);
	init();
	alert("Training is done, try classifying your handwriting!");
}

document.addEventListener('DOMContentLoaded', run);