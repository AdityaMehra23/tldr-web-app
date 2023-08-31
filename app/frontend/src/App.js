import './App.css';
import '@fontsource/roboto/300.css';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Stack from '@mui/material/Stack';

import AbstractGenerator from './components/AbstractGenerator/AbstractGenerator.js';
import { useState } from 'react';
import PatentSummarizer from './components/PatentSummarizer/PatentSummarizer';
import Navbar from './components/Navbar/Navbar';
function App() {

  let [useCase,changeUseCase] = useState('');

  const switchEditor = (value)=>{
    // console.log(value,useCase);
    if (value == null) {
      changeUseCase(useCase);
      return;
    }
    changeUseCase(value);
  }

  const renderEditors = () => {
    {
      console.log(useCase)
      switch (useCase) {
        case 'abstract':
          return <AbstractGenerator></AbstractGenerator>
        case 'patent':
          return <PatentSummarizer></PatentSummarizer>
      }
    }
  }

  return (
    <div className="App">
      <Navbar></Navbar>
      <Stack spacing={2} alignItems="center" style={{marginTop: '1rem'}}>
        <ToggleButtonGroup exclusive value={useCase} size="large" aria-label="options" onChange={(mouseEvent,value)=> switchEditor(value)}
        color='primary'>
          <ToggleButton value="patent" key="patent">
            Patent Summarizer
          </ToggleButton>
          <ToggleButton value="abstract" key="abstract">
            Abstract Generator
          </ToggleButton>
        </ToggleButtonGroup>
      </Stack>

    <div style={{marginTop: '2rem'}}>
     { renderEditors()}
    </div>
    

    </div>
  );
}

export default App;
