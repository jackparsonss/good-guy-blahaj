package main

import (
	"encoding/json"
	"net/http"

	"github.com/gorilla/websocket"
	"github.com/jackparsonss/good-guy-blahaj/data"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
)

var upgrader = websocket.Upgrader{}

func DataHandler(c echo.Context) error {
	data := data.GetData()

	return c.JSON(http.StatusOK, data)
}

func WSHandler(c echo.Context) error {
	upgrader.CheckOrigin = func(r *http.Request) bool { return true }
	ws, err := upgrader.Upgrade(c.Response(), c.Request(), nil)
	if err != nil {
		return err
	}
	defer ws.Close()

	for {
		// Write
		data := data.GetData()
		d, err := json.Marshal(data)
		if err != nil {
			c.Logger().Error(err)
		}

		err = ws.WriteMessage(websocket.TextMessage, d)
		if err != nil {
			c.Logger().Error(err)
		}
	}
}

func main() {
	e := echo.New()
	e.Use(middleware.Secure())
	e.Use(middleware.Logger())
	e.Use(middleware.Recover())

	e.GET("/", DataHandler)
	e.GET("/ws", WSHandler)

	e.Logger.Fatal(e.Start(":8080"))
}
