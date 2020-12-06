/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 * @flow strict-local
 */

import React from 'react';
import {
  SafeAreaView,
  StyleSheet,
  ScrollView,
  View,
  Text,
  StatusBar,
  LogBox
} from 'react-native';

import {
  Header,
  LearnMoreLinks,
  Colors,
  DebugInstructions,
  ReloadInstructions,
} from 'react-native/Libraries/NewAppScreen';

// import LoadMnist from './components/new_mnist_approach';
// import LoadCIFAR from './components/cifar_load';
// import LoadImageNet from './components/imagenet_load';
import SplitNet from './tfjs_implementations/final_implementation';
// import SplitNet from './tfjs_implementations/two_split_training';
// import CompleteNet from './tfjs_implementations/complete_network_resnet';
import io from 'socket.io-client';


const App: () => React$Node = () => {
  const server_url = '....';
  const socket = new io(server_url, {
    query: 'b64=1',
    pingTimeout: 360000
  });
  return (
    <>
      <SplitNet sock={socket} />
    </>
  );
};

export default App;
