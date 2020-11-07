import React from 'react';
import {
  StyleSheet,
  Text,
  View,
  Button,
  TouchableOpacity,
  StatusBar,
  Image,
  ActivityIndicator,
  Alert,
} from 'react-native';
import * as tf from '@tensorflow/tfjs';
const Realm = require('realm');
const train1 = require('../../assets/imagenet_1.json');

export default function LoadImageNet() {
  const [realmState, updateRealm] = React.useState(false);
  const IMAGE_H = 224;
  const IMAGE_W = 224;
  let NUM_DATA_ELEMENTS = 0;
  const ImageSchema = {
    name: 'Image',
    properties: {
      uid: 'int',
      data: 'string',
      label: 'int',
    },
  };

  function generateImageNetData(dataJson) {
    const dataImages = [];
    const dataLabels = [];

    dataJson.forEach((item, index) => {
      dataImages.push(JSON.stringify(item.image));
      dataLabels.push(item.label);
    });
    console.log(dataImages.length)
    NUM_DATA_ELEMENTS = dataImages.length;

    const data = {
      Images: dataImages,
      Labels: dataLabels,
    };
    return data;
  }

  function addToRealm() {
    const jsonData = generateImageNetData(train1);
    // const jsonData = [];
    Realm.open({schema: [ImageSchema]})
      .then((realm) => {
        for (let i = 0; i < NUM_DATA_ELEMENTS; i++) {
          console.log('Processing image: ' + String(i));
          realm.write(() => {
            const myImage = realm.create('Image', {
              uid: i,
              data: jsonData.Images[i],
              label: jsonData.Labels[i],
            });
          });
        }

        //verify the addition
        let images = realm.objects('Image');
        console.log(images.length);

        //update state
        updateRealm(true);

        //close instance
        realm.close();
      })
      .catch((error) => {
        console.log(error);
      });
  }

  function testRealm() {
    return;
  }

  async function _wrapper() {
    await tf.ready();
    addToRealm();
  }

  return (
    <View style={{padding: 50}}>
      <Button title={'Transfer to Realm'} onPress={() => _wrapper()} />
      <Text>Loading Data...</Text>
    </View>
  );
}
