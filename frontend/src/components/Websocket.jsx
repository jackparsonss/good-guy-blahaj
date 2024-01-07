import React, { useRef, useState, useEffect } from 'react';
import TypeWriter from "./TypeWriter.jsx"

const Websocket = () => {
  const connection = useRef(null);
  const [modified, setModified] = useState("");
  const [original, setOriginal] = useState("");

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

      setModified(m)
      setOriginal(o)
    })

    connection.current = socket;

    return () => connection.close();
  }, []);

  return (
    <div>
        <div className="text-lg text-white">
          <TypeWriter text={modified} delay={50} />
      </div>

      <div className="text-lg text-gray-400">
          <TypeWriter text={original} delay={50} />
        <p className="inline text-red-300"></p>
      </div>
    </div>
  )

};

export default Websocket;
