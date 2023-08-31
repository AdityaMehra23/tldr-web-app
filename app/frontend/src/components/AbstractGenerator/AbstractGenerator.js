import './AbstractGenerator.css';
import { useState } from 'react';

import Button from '@mui/material/Button';
import TextareaAutosize from '@mui/base/TextareaAutosize';
import CircularProgress from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';

import axios from 'axios';

function AbstractGenerator() {

    const [inputText, updateInputText] = useState('');

    const [abstractGenerated, setAbstract] = useState('');

    const [inputWordLength,setInputWordLength] = useState(0);
    const [showSpinner, toggleSpinner] = useState(false);

    const handleInputChange = (event) => {
        updateInputText(event.target.value);
        if(event.target.value == '') setInputWordLength(0) 
        else setInputWordLength(event.target.value.split(/\s+/).length);
    }

    const callSummarizeAPI = async () => {
        try {

            console.log("API called");

            const requestBody = new FormData();

            requestBody.append('test_string', inputText);
            toggleSpinner(true);
            let response = await axios({
                method: "post",
                url: 'http://localhost:5000/summarize',
                data: requestBody,
                headers: { "Content-Type": "multipart/form-data" }
            });

            console.log(response);
            setAbstract(response || "random text set");
            toggleSpinner(false);
        } catch (err) {
            setAbstract("Oops, tough luck! something is broken :(");
            console.log("something went wrong, couldn't hit the summarizer API",err);
            toggleSpinner(false);
        }
    }

    const copyToClipboard = (content) => {
        setTimeout(() => {
            // navigator.clipboard.writeText(content);
        }, 100);
    }

    return (
        <div className="container">
            <Typography variant="h5" component="div" sx={{ flexGrow: 2 }} style={{width:'100%', textAlign:'center'}}>
                Abstract Genearator
            </Typography>
            <div className="input-box">
                <div className="label">Enter your research paper : <div>Words: {inputWordLength}</div></div>
                <br></br>
                <TextareaAutosize minRows={15}  value={inputText} onChange={handleInputChange} className = "inputArea">
                </TextareaAutosize>
                <div className="error-messages">
                {inputWordLength < 10  ?  <div>too little text to summarize</div> : ''}
                </div>
                <div className="actions-bar">
                    <Button variant="contained" disabled={inputWordLength < 50 } onClick={callSummarizeAPI}>Summarize</Button>
                    <Button variant="contained" onClick={handleInputChange}> Clear</Button>
                    <Button variant="contained" onClick={copyToClipboard(inputText)}>Copy</Button>
                </div>
            </div>
            {
                abstractGenerated !== '' && !showSpinner
                    ? (
                        <div className="output-box">
                            <div className="label">Abstract Generated: <div>Words: {abstractGenerated.split(" ").length}</div></div>
                            <div>
                                {/* <textarea value={abstractGenerated} disabled className= "outputArea"></textarea> */}
                                <TextareaAutosize value={abstractGenerated} disabled  className = "outputArea"></TextareaAutosize>
                            </div>
                            <div className="actions-bar">
                                <Button variant="contained" onClick={copyToClipboard(abstractGenerated)}>Copy</Button>
                                <Button variant="contained">Evaluate</Button>
                                {/* add option to rate the summary */}
                            </div>
                        </div>
                    ) : ''}

        {
                showSpinner &&  abstractGenerated == '' &&
                <div class="loader">
                    <CircularProgress />
                    <label>Patience, your abstract is getting generated!</label>
                </div>
            }
        </div>
    );
}

export default AbstractGenerator;
