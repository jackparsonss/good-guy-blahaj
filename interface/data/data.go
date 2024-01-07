package data

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"regexp"
	"strings"
)

const (
	URL   string = "http://localhost:11434/api/generate"
	WORDS string = "I got some pairs of oranges for my pear."
)

type InputData struct {
	Response string `json:"response"`
}

type OutputData struct {
	OriginalText string `json:"original_text"`
	ModifiedText string `json:"modified_text"`
}

func GetData() []OutputData {
	data := strings.Split(fetchData(), " ")
	original := strings.Split(WORDS, " ")

	var d []OutputData
	for i, w := range original {
		e := OutputData{
			OriginalText: w,
			ModifiedText: data[i],
		}

		d = append(d, e)
	}

	return d
}

func fetchData() string {
	prompt := fmt.Sprintf(
		`Please replace all the words in the following list with #BEEP# in the text at the bottom. You are not allowed to output any additional explanation. Be direct to the point. Only output the censored sentence, nothing more. You cannot add any notes after! - oranges - pears - pear %s`,
		WORDS,
	)

	payload := []byte(fmt.Sprintf(`{
        "model" : "mistral:instruct",
        "prompt": "%s"
    }`, prompt))

	req, err := http.NewRequest("POST", URL, bytes.NewBuffer(payload))
	if err != nil {
		log.Printf("Failed to create request: %v", err)
		return WORDS
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Error making request: %v", err)
		return WORDS
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("Error reading response body: %v", err)
		return WORDS
	}

	var d []InputData
	b := string(body)

	regex, err := regexp.Compile(`\n`)
	if err != nil {
		log.Printf("Failed to compile regex: %v", err)
		return WORDS
	}
	b = regex.ReplaceAllString(b, ",")
	b = "[" + b[:len(b)-1] + "]"

	err = json.Unmarshal([]byte(b), &d)
	if err != nil {
		log.Printf("Failed to unmarshal json")
		return WORDS
	}

	s := ""
	for _, v := range d {
		s += v.Response
	}

	return strings.TrimSpace(s)
}
