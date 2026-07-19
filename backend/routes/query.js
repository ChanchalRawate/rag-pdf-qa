const express = require("express");
const axios = require("axios");

const router = express.Router();

router.post("/", async (req, res) => {
  try {
    const { question } = req.body;

    const response = await axios.post(
      "http://127.0.0.1:8000/query",
      {
        question,
      }
    );

    res.json({
      answer: response.data.answer,
    });
  } catch (error) {
    console.error(error);

    res.status(500).json({
      answer: "Error communicating with Python service",
    });
  }
});

module.exports = router;