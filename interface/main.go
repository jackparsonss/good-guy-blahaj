package main

import (
	"encoding/json"
	"net"
	"net/http"

	"github.com/gorilla/websocket"
	"github.com/jackparsonss/good-guy-blahaj/data"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
)

var upgrader = websocket.Upgrader{}

const orca string = "orca.mami2.moe:9033"

type Data struct {
	Original   string `json:"original"`
	IsCensored bool   `json:"is_censored"`
}

func DataHandler(c echo.Context) error {
	data := data.GetData()

	return c.JSON(http.StatusOK, data)
}

func WSHandler(c echo.Context) error {
	upgrader.CheckOrigin = func(r *http.Request) bool { return true }
	ws, err := upgrader.Upgrade(c.Response(), c.Request(), nil)
	if err != nil {
		c.Logger().Error(err)
		return err
	}
	defer ws.Close()

	connection, err := net.Dial("tcp", orca)
	if err != nil {
		c.Logger().Error(err)
	}
	defer connection.Close()

	for {
		_, err = connection.Write([]byte("i'm gonna push you off the cliff"))
		if err != nil {
			c.Logger().Error(err)
			return err
		}

		buffer := make([]byte, 32768)
		mLen, err := connection.Read(buffer)
		if err != nil {
			c.Logger().Error(err)
			return err
		}

		var data []data.Data
		err = json.Unmarshal(buffer[:mLen], &data)
		if err != nil {
			c.Logger().Error(err)
			return err
		}

		var d []Data
		for _, w := range data {
			e := Data{
				IsCensored: w.IsCensored,
				Original:   w.Original,
			}

			d = append(d, e)
		}

		output, err := json.Marshal(d)
		if err != nil {
			c.Logger().Error(err)
			return err
		}

		err = ws.WriteMessage(websocket.TextMessage, output)
		if err != nil {
			c.Logger().Error(err)
			return err
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
