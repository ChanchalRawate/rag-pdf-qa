const express = require("express");
const multer = require("multer");
const axios = require("axios");
const path = require("path");

const router = express.Router();

const storage = multer.diskStorage({
  destination: "./uploads",
  filename: (req, file, cb) => {
    cb(null, file.originalname);
  },
});

const upload = multer({ storage });

router.post("/", upload.single("pdf"), async (req, res) => {
   console.log("✅ Upload route called");

  try {
    // Absolute path of uploaded PDF
    const pdfPath = path.join(__dirname, "..", "uploads", req.file.originalname);

       console.log("📄 PDF Path:", pdfPath);
       console.log("📤 Calling Python...");

    // Tell FastAPI to process the PDF
    await axios.post("http://127.0.0.1:8000/process-pdf", {
      path: pdfPath,
    });
    console.log("✅ Python processed the PDF");

    res.json({
      success: true,
      message: "PDF uploaded and processed successfully!",
    });
  } catch (error) {
    console.error(error);

    res.status(500).json({
      success: false,
      message: "Failed to process PDF.",
    });
  }
});

module.exports = router;