package main

import (
	"net/http"

	"github.com/jackparsonss/good-guy-blahaj/data"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
)

func DataHandler(c echo.Context) error {
	data := data.GetData()

	return c.JSON(http.StatusOK, data)
}

func main() {
	e := echo.New()
	e.Use(middleware.Secure())
	e.Use(middleware.Logger())
	e.Use(middleware.Recover())

	e.GET("/", DataHandler)

	e.Logger.Fatal(e.Start(":8080"))
}
