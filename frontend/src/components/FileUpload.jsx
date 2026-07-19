import axios from "axios";

const API = import.meta.env.VITE_API_URL;

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
      const response = await axios.post(`${API}/upload-pdf`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      console.log(response.data);
      alert("PDF uploaded successfully!");
    } catch (error) {
      console.error(error);
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
