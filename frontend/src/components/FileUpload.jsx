import axios from "axios";

const API = "http://localhost:8080";

function FileUpload({ file, setFile }) {
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a PDF first.");
      return;
    }

    const formData = new FormData();
    formData.append("pdf", file);

    try {
      const token = localStorage.getItem("token");

      const response = await axios.post(`${API}/upload-pdf`, formData, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "multipart/form-data",
        },
      });

      console.log("Upload response:", response.data);
      alert("PDF uploaded successfully!");
    } catch (error) {
      console.error("Upload error:", error);

      if (error.response) {
        console.error("Status:", error.response.status);
        console.error("Response:", error.response.data);
      }

      alert("Upload failed.");
    }
  };

  return (
    <div>
      <h2>Upload Document</h2>

      <div className="drop-zone">
        <p>📄</p>
        <p>Drag & Drop PDF Here</p>

        <input
          type="file"
          id="pdf-upload"
          accept=".pdf"
          hidden
          onChange={handleFileChange}
        />

        <label htmlFor="pdf-upload" className="upload-btn">
          Select File
        </label>

        <button className="upload-btn" onClick={handleUpload}>
          Upload PDF
        </button>
      </div>
    </div>
  );
}

export default FileUpload;
