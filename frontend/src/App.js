import React, { useState } from "react";
import "./App.css";

function App() {
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [finalMessage, setFinalMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);

  const handleSend = async () => {
    if (!message.trim()) return;
    
    setIsLoading(true);
    const userMessage = message;
    setMessage("");
    
    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userMessage }),
      });
      const data = await res.json();
      setResponse(data.predicted_intent);
      setShowFeedback(true);
      
      // Add to chat history
      setChatHistory(prev => [...prev, {
        type: 'user',
        message: userMessage,
        timestamp: new Date()
      }, {
        type: 'bot',
        message: data.predicted_intent,
        confidence: data.confidence,
        timestamp: new Date()
      }]);
    } catch (error) {
      console.error('Error:', error);
      setFinalMessage("Sorry, there was an error processing your request. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCorrect = () => {
    const lastBotMessage = chatHistory[chatHistory.length - 1];
    const intentName = lastBotMessage?.message || response;
    setFinalMessage(`I believe you need assistance in "${intentName}". Let me take care of that! ✅`);
    setShowFeedback(false);
  };

  const handleIncorrect = async () => {
    const lastBotMessage = chatHistory[chatHistory.length - 1];
    const confidence = lastBotMessage?.confidence || 0;
    const intentName = lastBotMessage?.message || response;
    const userMessage = chatHistory[chatHistory.length - 2]?.message || message;
    
    try {
      await fetch("http://127.0.0.1:8000/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: userMessage,
          predicted_intent: intentName,
          correct: false,
          true_label: null // Human agents can fill this later
        }),
      });
    } catch (error) {
      console.error('Error logging feedback:', error);
    }

    // Determine response based on confidence level
    let responseMessage;
    if (confidence >= 0.9) {
      responseMessage = `I have ${Math.round(confidence * 100)}% confidence on your request type "${intentName}" but if I am wrong, please tell me the issue again for clarification. 🤔`;
    } else if (confidence >= 0.7) {
      responseMessage = `I have ${Math.round(confidence * 100)}% confidence on your request type "${intentName}" but if I am wrong, please tell me the issue again for clarification. 🤔`;
    } else {
      responseMessage = "I am transferring you to a human agent. Thank you for your patience! 🤝";
    }
    
    setFinalMessage(responseMessage);
    setShowFeedback(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    setChatHistory([]);
    setResponse(null);
    setShowFeedback(false);
    setFinalMessage("");
  };

  return (
    <div className="app">
      <div className="header">
        <div className="header-content">
          <div className="logo">
            <span className="airplane-icon">✈️</span>
            <h1>Airline Support Assistant</h1>
          </div>
          <p className="subtitle">Your AI-powered customer service companion</p>
        </div>
      </div>

      <div className="main-container">
        <div className="chat-container">
          {chatHistory.length > 0 && (
            <div className="chat-history">
              {chatHistory.map((item, index) => (
                <div key={index} className={`message ${item.type}`}>
                  <div className="message-content">
                    {item.type === 'user' ? (
                      <div className="user-message">
                        <div className="message-text">{item.message}</div>
                        <div className="message-time">
                          {item.timestamp.toLocaleTimeString()}
                        </div>
                      </div>
                    ) : (
                      <div className="bot-message">
                        <div className="bot-avatar">🤖</div>
                        <div className="message-details">
                          <div className="intent-label">Predicted Intent:</div>
                          <div className="intent-value">{item.message}</div>
                          {/* <div className="confidence">
                            Confidence: {Math.round(item.confidence * 100)}%
                          </div> */}
                          <div className="message-time">
                            {item.timestamp.toLocaleTimeString()}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {isLoading && (
            <div className="loading-container">
              <div className="loading-spinner"></div>
              <p>Analyzing your request...</p>
            </div>
          )}

          {response && showFeedback && (
            <div className="feedback-container">
              <div className="feedback-card">
                <h3>Was this prediction correct?</h3>
                <div className="feedback-buttons">
                  <button 
                    className="feedback-btn correct-btn" 
                    onClick={handleCorrect}
                  >
                    ✅ Yes, Correct
                  </button>
                  <button 
                    className="feedback-btn incorrect-btn" 
                    onClick={handleIncorrect}
                  >
                    ❌ No, Incorrect
                  </button>
                </div>
              </div>
            </div>
          )}

          {finalMessage && (
            <div className="final-message">
              <div className="final-message-card">
                <p>{finalMessage}</p>
                <button className="new-chat-btn" onClick={clearChat}>
                  Start New Conversation
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="input-container">
          <div className="input-wrapper">
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me anything about your flight, booking, baggage, or any airline service..."
              className="message-input"
              rows="2"
              disabled={isLoading}
            />
            <button 
              onClick={handleSend} 
              className="send-btn"
              disabled={isLoading || !message.trim()}
            >
              <span className="send-icon">📤</span>
            </button>
          </div>
          <div className="input-hint">
            Press Enter to send, Shift+Enter for new line
          </div>
        </div>
      </div>

      <div className="footer">
        <div className="footer-content">
          <p>Powered by DistilBERT AI • 28+ Intent Categories • Real-time Feedback</p>
        </div>
      </div>
    </div>
  );
}

export default App;
