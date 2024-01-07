import React, { useRef, useState, useEffect } from 'react';
import TypeWriter from "./TypeWriter.jsx"

const Websocket = () => {
  const connection = useRef(null);
  const [modified, setModified] = useState("");
  const [original, setOriginal] = useState("");
  const [isCensored, setIsCensored] = useState(false);

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8080/ws");

    // Connection opened
    socket.addEventListener("open", (event) => {
      console.log("Connection established");
    })

    // Listen for messages
    socket.addEventListener("message", (event) => {
      console.log("Got message: ", event.data)
      let m = ""
      let o = ""
      const data = JSON.parse(event.data)
      data.forEach(el => {
        m += el.is_censored ? "#BLEEP# " : el.original + " "
        o += el.original + " "
      })

      setModified(m.trimEnd() + ".\n")
      setOriginal(o.trimEnd() + ".\n")
    })

    connection.current = socket;

    return () => connection.close();
  }, []);

  const toggleSwitch = () => {
    setIsCensored(!isCensored);
  };

  return (
    <div>
      <div className="relative inline-block w-12 h-6 bg-gray-300 rounded-full cursor-pointer" onClick={toggleSwitch}>
      <div
        className={`absolute w-6 h-6 bg-white rounded-full shadow-md transform transition-transform duration-300 ${
          isCensored ? 'translate-x-full' : 'translate-x-0'
        }`}
      />
      </div>
        <div>Censor</div>
      <div className="text-lg text-white">
        {isCensored && <div>{modified}</div>}
        {!isCensored && <div>{original}</div>}

        {/* {isCensored && <TypeWriter text={modified} delay={50} />} */}
        {/* {!isCensored && <TypeWriter text={original} delay={50} />} */}
      </div>
    </div>
  )
};

export default Websocket;
