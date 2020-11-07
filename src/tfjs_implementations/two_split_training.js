import React from 'react';
import {Text, View, Button} from 'react-native';
import * as tf from '@tensorflow/tfjs';
const Realm = require('realm');
import {Table, Row, Rows} from 'react-native-table-component';
import {bundleResourceIO} from '@tensorflow/tfjs-react-native';
//import LoadMnist from '../components/init_mnist';

export default function SplitNet(props) {
  //table display
  const HeadTable = ['EPOCHS', 'BATCH', 'LOSS', 'ACC.', 'WALL CLOCK TIME'];
  const DataTable = [];
  const [trainingState, updateTrainingStuff] = React.useState(DataTable);

  //states to monitor training
  let EPOCH_COUNTER = 0;
  let BATCH_COUNTER = -1;
  let OFFSET = 0;

  //states to monitor data transmission
  let sentOutputsToB = false;
  let receivedGradsFromB = false;

  //states to monitor current batch images and labels
  let batchId = null;
  let batchImages = null;
  let batchLabels = null;

  //models and optimizers
  let modelA = null;
  let modelC = null;
  let optimA = null;
  let optimC = null;
  let outA = null;

  //constants
  const NUM_TRAIN = 10;
  const NUM_TEST = 10;
  const BATCH_SIZE = 1;
  //const TRAIN_BATCHES = NUM_TRAIN / BATCH_SIZE;
  const TOTAL_TRAIN_BATCHES = 90;
  const TOTAL_TEST_BATCHES = NUM_TEST / BATCH_SIZE;
  const TOTAL_EPOCHS = 1;
  var global_start = 0;
  var global_end = 0;
  //server connection credentials
  const server_url = 'http://13.230.90.9:8080';

  async function stateSocketHandler(socket_val) {
    if (socket_val === 'output_to_B') {
      //gradSocketCalled = true;
      console.log('SOCKET_MESSAGE: Gradients from B received!');
      await _receiveGradFromB();
      //restart the training
      const newDataEntry = [
        EPOCH_COUNTER,
        BATCH_COUNTER,
        String(0.889),
        String(88.9),
        String(1235),
      ];
      const currState = trainingState;
      currState.push(newDataEntry);
      updateTrainingStuff(currState);

      //dispose older data
      tf.dispose(batchImages);
      tf.dispose(batchLabels);
      console.log('Memory Stats: Finish Training');
      console.log(tf.memory());
      console.log(
        'EPOCH: ' +
          EPOCH_COUNTER +
          ' BATCH: ' +
          BATCH_COUNTER +
          ' completed...',
      );
      await _forwardToB();
    }
  }

  const socket = props.sock;

  //socket listeners
  socket.on('connect', (socket_data) => {
    console.log(socket_data);
    console.log('Connected to the server!');
    socket.on('state', (socket) => {
      stateSocketHandler(socket).then((r) => console.log(r));
    });
  });

  socket.on('disconnect', (socket_data) => {
    console.log(socket_data);
    console.log('Disconnected from the server!');
  });

  socket.on('reconnecting', (socket_data) => {
    console.log(socket_data);
    console.log('Trying to reconnect!');
  });

  socket.on('connect_timeout', (timeout) => {
    console.log(timeout);
    console.log('Timed out! Huh?');
  });

  //constants
  const IMAGE_H = 299;
  const IMAGE_W = 299;

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
        console.log("Label Array Length", labels.length)
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

  //a method to receive gradients from B to A and perform remaining computations on the client side

  //a method used from receiving and processing outputs from splitB on receiving a socket signal
  async function _receiveGradFromB() {

    const request_url = server_url + '/outputToB';
    //make a GET request to receive the gradients
    const request = new Request(request_url, {
      method: 'GET',
    });

    if (receivedGradsFromB === false) {
      const new_start = new Date().getTime();
      const gradFromB = await fetch(request);
      const new_end = new Date().getTime();
      console.log("GET Request Time", (new_end - new_start)/1000)
      var start = new Date().getTime();
      const response = await gradFromB.json();
      const gradB = response.inputs;
      const dJ_dh = tf.tensor(gradB);

    
      const targetA = tf.sub(outA, dJ_dh);
      // const loss = tf.losses.meanSquaredError(targetA, outA, tf.Reduction.SUM);
      modelA.fit(batchImages, targetA);
      var end = new Date().getTime();
      global_end = new Date().getTime();
      console.log('Time for Backward Pass in A', (end - start) / 1000);
      console.log(
        'Overall Time(s) for spilt learning for 1 batch',
        (global_end - global_start) / 1000,
      );
      // console.log('modelA');
      // modelA.getWeights()[0].print();
      receivedGradsFromB = true;
      sentOutputsToB = false;

      //remove tensors from memory
      tf.dispose(dJ_dh);
    }
    //FIXME: what about the optimizer again? need to test it!
  }

  //a method used for sending outputs to splitB computed locally by SplitA
  async function _forwardToB() {
    //show memory stats
    //if batches are complete, next epoch
    if (BATCH_COUNTER >= TOTAL_TRAIN_BATCHES) {
      //inc the epoch count
      EPOCH_COUNTER++;
      //if epochs have reached num epochs, stop training
      if (EPOCH_COUNTER >= TOTAL_EPOCHS) {
        console.log('Congratulations! Training Complete');
        console.log('NUM EPOCHS:', EPOCH_COUNTER);
        console.log('NUM BATCHES:', BATCH_COUNTER);
        return;
      }
      BATCH_COUNTER++;
      OFFSET = 0;
    }

    //increase the batch counter and set a new offset
    BATCH_COUNTER++;
    OFFSET = BATCH_COUNTER * BATCH_SIZE;

    global_start = new Date().getTime()
    var start = new Date().getTime();
    //prepare data
    const data = await prepareBatchData(BATCH_SIZE, OFFSET);
    batchId = data.id;
    batchImages = data.inputImages;
    batchLabels = data.inputLabels;

    //start training for the batch
    console.log('Running Epochs:', EPOCH_COUNTER, ', Batch:', BATCH_COUNTER);
    //compute outputs from modelA
    outA = await modelA.apply(batchImages);

    const sendA = outA.arraySync();
    //const jsonA = {inputs: outA.arraySync()};
    var end = new Date().getTime();

    console.log("Processing Time(s) for Forward Pass in A", (end-start)/1000);

    const request_url = server_url + '/outputToB';
    if (sentOutputsToB === false) {
      const data = JSON.stringify({inputs: sendA});
      const new_start = new Date().getTime();
      fetch(request_url, {
        method: 'POST',
        body: data,
        headers: {
          'Content-Type': 'application/json',
        },
      })
        .then(function (response) {
          const new_end = new Date().getTime();
          console.log("POST Response Time", (new_end-new_start)/1000);
        });
        // .catch(function(error){
        //   console.log(error)
        // });
      console.log('POST - A to B');
      sentOutputsToB = true;
      receivedGradsFromB = false;

      // //clean tensor
      // tf.dispose(outA);
    }
  }

  async function _define() {
    const NUM_CHANNELS = 3;

    const modelAjson = require('../../assets/modelAthird/model.json');
    const modelAweights = require('../../assets/modelAthird/group1-shard1of1.bin');
    const modelA = await tf.loadLayersModel(
      bundleResourceIO(modelAjson, modelAweights),
    );
 
    
    const optimA = tf.train.adam(0.0001);

    modelA.compile({optimizer: optimA, loss: 'meanSquaredError'});
    
    const finalObjects = {
      splitA: modelA,
      optimA: optimA,
    };
    return finalObjects;
  }

  //main training wrapper
  async function _train() {
    //initiate tf
    await tf.ready();

    //initiate models and optimizers
    const models = await _define();
    modelA = models.splitA;
    optimA = models.optimA;

    console.log('Begin Training...');
    await _forwardToB();
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
      <Table borderStyle={{borderWidth: 1, borderColor: '#000'}}>
        <Row
          data={HeadTable}
          style={{
            height: 50,
            alignContent: 'center',
            backgroundColor: '#ececec',
          }}
          textStyle={{
            margin: 10,
            color: '#000',
            fontSize: 17,
            fontWeight: 'bold',
          }}
        />
        <Rows
          data={trainingState}
          textStyle={{
            margin: 10,
            color: '#000',
            fontSize: 15,
          }}
        />
      </Table>
    </View>
  );
}
