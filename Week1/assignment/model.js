(async function() {
    const trainingUrl = 'wdbc-train.csv';

    // Take a look at the 'wdbc-train.csv' file and specify the column
    // that should be treated as the label in the space below.
    // HINT: Remember that you are trying to build a classifier that 
    // can predict from the data whether the diagnosis is malignant or benign.
    const trainingData = tf.data.csv(trainingUrl, {

        // YOUR CODE HERE
        columnConfigs: {
            diagnosis: {
                isLabel: true
            }
        }

    });

    // Convert the training data into arrays in the space below.
    // Note: In this case, the labels are integers, not strings.
    // Therefore, there is no need to convert string labels into
    // a one-hot encoded array of label values like we did in the
    // Iris dataset example. 
    const convertedTrainingData = await trainingData.map(({
        xs,
        ys
    }) => {
        console.log(Object.values(xs));
        return ({
            xs: Object.values(xs),
            ys: Object.values(ys.diagnosis)
        })
    }).batch(10); // YOUR CODE HERE

    console.log(await convertedTrainingData)
    const testingUrl = 'wdbc-test.csv';

    // Take a look at the 'wdbc-test.csv' file and specify the column
    // that should be treated as the label in the space below..
    // HINT: Remember that you are trying to build a classifier that 
    // can predict from the data whether the diagnosis is malignant or benign.
    const testingData = tf.data.csv(testingUrl, {

        // YOUR CODE HERE
        columnConfigs: {
            diagnosis: {
                isLabel: true
            }
        }

    });

    // Convert the testing data into arrays in the space below.
    // Note: In this case, the labels are integers, not strings.
    // Therefore, there is no need to convert string labels into
    // a one-hot encoded array of label values like we did in the
    // Iris dataset example. 
    const convertedTestingData = await testingData.map(({
        xs,
        ys
    }) => {
        console.log(Object.values(xs));
        return ({
            xs: Object.values(xs),
            ys: Object.values(ys.diagnosis)
        })
    }).batch(10); // YOUR CODE HERE


    // Specify the number of features in the space below.
    // HINT: You can get the number of features from the number of columns
    // and the number of labels in the training data. 
    const numOfFeatures = (await trainingData.columnNames()).length - 1 // YOUR CODE HERE
    console.log(numOfFeatures)

    // In the space below create a neural network that predicts 1 if the diagnosis is malignant
    // and 0 if the diagnosis is benign. Your neural network should only use dense
    // layers and the output layer should only have a single output unit with a
    // sigmoid activation function. You are free to use as many hidden layers and
    // neurons as you like.  
    // HINT: Make sure your input layer has the correct input shape. We also suggest
    // using ReLu activation functions where applicable. For this dataset only a few
    // hidden layers should be enough to get a high accuracy.  
    const model = tf.sequential();
    // YOUR CODE HERE
    model.add(tf.layers.dense({
        inputShape: [numOfFeatures],
        activation: "relu",
        units: 30
    }));
    model.add(tf.layers.dense({
        activation: "relu",
        units: 16
    }));
    model.add(tf.layers.dense({
        activation: "relu",
        units: 8
    }));
    model.add(tf.layers.dense({
        activation: "relu",
        units: 4
    }));
    model.add(tf.layers.dense({
        activation: "relu",
        units: 2
    }));
    model.add(tf.layers.dense({
        activation: "sigmoid",
        units: 1
    }));

    // Compile the model using the binaryCrossentropy loss, 
    // the rmsprop optimizer, and accuracy for your metrics. 
    model.compile({
        loss: "binaryCrossentropy",
        optimizer: tf.train.rmsprop(0.07)
    }); // YOUR CODE HERE


    await model.fitDataset(convertedTrainingData, {
        epochs: 100,
        validationData: convertedTestingData,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                console.log("Epoch: " + epoch + " Loss: " + logs.loss + " Accuracy: " + logs.acc);
            }
        }
    });
    await model.save('downloads://my_model');
})()