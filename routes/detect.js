const express = require("express");
const router = express.Router();
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");
const path = require("path");

// Ensure uploads directory exists
const uploadsDir = "uploads";
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
}

router.get("/", async (req, res) => {
    res.render("detection/detect.ejs");
});

// Multer setup for file uploads with file type validation
const upload = multer({ 
    dest: "uploads/",
    limits: {
        fileSize: 16 * 1024 * 1024 // 16MB limit
    },
    fileFilter: (req, file, cb) => {
        const allowedTypes = /jpeg|jpg|png|gif|bmp|tiff/;
        const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
        const mimetype = allowedTypes.test(file.mimetype);
        
        if (mimetype && extname) {
            return cb(null, true);
        } else {
            cb(new Error('Only image files are allowed'));
        }
    }
});

router.post("/", upload.single("uploadedImage"), async (req, res) => {
    let tempFilePath = null;
    
    try {
        console.log("=== Request Debug Info ===");
        console.log("File received:", req.file ? req.file.originalname : "No file");
        console.log("File path:", req.file ? req.file.path : "No path");
        console.log("File size:", req.file ? req.file.size : "No size");
        
        // Check if file was uploaded
        if (!req.file) {
            return res.status(400).json({ 
                error: "No image file uploaded",
                details: "Please select an image file to upload"
            });
        }
        
        tempFilePath = req.file.path;
        
        // Verify file exists
        if (!fs.existsSync(tempFilePath)) {
            throw new Error("Uploaded file not found on server");
        }
        
        // Create FormData for Python API - use 'image' as expected by Python
        const formData = new FormData();
        formData.append("image", fs.createReadStream(tempFilePath), {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });
        
        console.log("Sending request to Python API...");
        
        // Send image to Python Flask API with timeout
        const response = await axios.post("http://localhost:5000/predict", formData, {
            headers: {
                ...formData.getHeaders(),
                'Accept': 'application/json'
            },
            timeout: 60000, // 60 seconds timeout
            maxContentLength: 16 * 1024 * 1024,
            maxBodyLength: 16 * 1024 * 1024
        });
        
        console.log("Python API Response Status:", response.status);
        console.log("Python API Response Data:", response.data);
        
        // Validate response from Python API
        if (!response.data) {
            throw new Error("Empty response from Python API");
        }
        
        if (response.data.error) {
            throw new Error(`Python API Error: ${response.data.error}`);
        }
        
        if (!response.data.predicted_disease) {
            throw new Error("Invalid response format: missing predicted_disease");
        }
        
        // Send successful response
        res.json({
            success: true,
            predicted_disease: response.data.predicted_disease,
            confidence_score: response.data.confidence_score,
            cure_and_precaution: response.data.cure_and_precaution,
            source: response.data.source || "AI Model",
            status: response.data.status || "success"
        });
        
    } catch (err) {
        console.error("=== Error Details ===");
        console.error("Error:", err.message);
        console.error("Stack:", err.stack);
        
        // Handle specific error types
        let errorMessage = "Prediction failed";
        let statusCode = 500;
        
        if (err.code === 'ECONNREFUSED') {
            errorMessage = "Python API server is not running";
            statusCode = 503;
        } else if (err.code === 'ENOENT') {
            errorMessage = "File processing error";
            statusCode = 400;
        } else if (err.response) {
            // Axios error with response
            errorMessage = `API Error: ${err.response.status} - ${err.response.data?.error || err.response.statusText}`;
            statusCode = err.response.status;
        } else if (err.message.includes('timeout')) {
            errorMessage = "Request timeout - please try again";
            statusCode = 408;
        }
        
        res.status(statusCode).json({ 
            error: errorMessage,
            details: err.message,
            timestamp: new Date().toISOString()
        });
        
    } finally {
        // Clean up temporary file
        if (tempFilePath && fs.existsSync(tempFilePath)) {
            try {
                fs.unlinkSync(tempFilePath);
                console.log("Temporary file cleaned up:", tempFilePath);
            } catch (cleanupErr) {
                console.error("Failed to cleanup temp file:", cleanupErr.message);
            }
        }
    }
});

// Health check endpoint
router.get("/health", async (req, res) => {
    try {
        const response = await axios.get("http://localhost:5000/health", { timeout: 5000 });
        res.json({
            status: "healthy",
            python_api: response.data
        });
    } catch (err) {
        res.status(503).json({
            status: "unhealthy",
            error: "Python API not available",
            details: err.message
        });
    }
});

module.exports = router;