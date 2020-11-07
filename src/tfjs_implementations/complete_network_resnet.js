import React from 'react';
import {Text, View, Button} from 'react-native';
import * as tf from '@tensorflow/tfjs';
import {bundleResourceIO} from '@tensorflow/tfjs-react-native';
const Realm = require('realm');
//import LoadMnist from '../components/init_mnist';

export default function CompleteNet(){
  //table display
  const DataTable = [];
  const [] = React.useState(DataTable);

  //states to monitor training

  //states to monitor current batch images and labels
  let batchImages = null;
  let batchLabels = null;

  //models and optimizers
  let model = null;
  let optim = null;

  //constants
  const BATCH_SIZE = 1;
  //const TRAIN_BATCHES = NUM_TRAIN / BATCH_SIZE;
  const TOTAL_TRAIN_BATCHES = 20;
  const TOTAL_EPOCHS = 1;

  //constants
  const IMAGE_H = 224;
  const IMAGE_W = 224;

  //local Realm db object access schema
  const ImageSchema = {
    name: 'Image',
    properties: {
      uid: 'int',
      data: 'string',
      label: 'int',
    },
  };

  //loads data from each batch from the local database and returns a wrapper with id, inputTensor, labelTensor
  function prepareBatchData(batchSize, offSet) {
    const batchData = Realm.open({schema: [ImageSchema]})
      .then((realm) => {
        const start = offSet;
        const end = offSet + batchSize;
        const batch = realm
          .objects('Image')
          .filtered('(' + start + ' <= uid)&&(uid < ' + end + ')');

        //tensor generation
        const batch_js = [];
        const labels = [];
        for (let i = 0; i < batchSize; i++) {
          batch_js.push(JSON.parse(batch[i].data));
          labels.push(batch[i].label);
        }
        console.log(labels)
        const batch_tensor = tf.tensor4d(
          batch_js,
          [batchSize, IMAGE_H, IMAGE_W, 3],
          'float32',
        );
        const encoded_labels = tf.oneHot(tf.tensor1d(labels, 'int32'), 1000);
        const batchData = {
          id: offSet / 100,
          inputImages: batch_tensor,
          inputLabels: encoded_labels,
        };
        realm.close();
        return batchData;
      })
      .catch((error) => console.log(error));

    return batchData;
  }

  async function _define() {
    //define model a
    const modeljson = require('../../assets/DenseNet201/model.json');
    const modelweights = require('../../assets/DenseNet201/group1-shard1of1.bin');
    model = await tf.loadLayersModel(bundleResourceIO(modeljson, modelweights));
    // console.log(model.summary());
    optim = tf.train.adam(0.01);

    model.compile({optimizer: optim, loss: 'categoricalCrossentropy'});
    const finalObjects = {
      model: model,
      optim: optim,
    };
    return finalObjects;
  }

  //main training wrapper
  async function _train() {
    //initiate tf
    await tf.ready();

    //initiate models and optimizers
    const models = await _define();
    model = models.model;
    optim = models.optim;

    console.log('Begin Training...');
    var epoch;
    for (epoch = 0; epoch < TOTAL_EPOCHS; epoch++) {
      console.log(epoch, TOTAL_EPOCHS);
      var batch;
      for (batch = 0; batch < TOTAL_TRAIN_BATCHES; batch++) {
        var start = new Date().getTime();
        const OFFSET = batch*BATCH_SIZE;
        let data = await prepareBatchData(BATCH_SIZE, OFFSET);
        batchImages = data.inputImages;
        batchLabels = data.inputLabels;
        
        // const output = await model.predict(batchImages);
        const h = await model.fit(batchImages, batchLabels, {batchSize:1});
        console.log(h);

        var end = new Date().getTime();
        console.log('Time Taken', end - start);
      }
      console.log('EPOCH: ', epoch, ' over');
    }
  }
  return (
    <View style={{padding: 50}}>
      <View
        style={{
          alignItems: 'center',
          justifyContent: 'center',
          paddingBottom: 20,
        }}>
        <Text style={{fontSize: 32, fontWeight: 'bold'}}>SplitNN</Text>
      </View>
      <Button title={'Begin Training on MNIST data'} onPress={() => _train()} />
    </View>
  );
}
