const express = require("express");
const router = express.Router();

router.get("/alert", (req, res) => {
  res.render("weather/weatherAlert.ejs", { tempAlert: null });
});

router.post("/alert", async (req, res) => {
  try {
    let city = req.body.city;
    
    let response = await fetch(
      `${process.env.API_URL}?q=${city}&appid=${process.env.API_KEY}&units=metric`
    );
    let jsonRes = await response.json();
    let resObject = {
      city: city,
      temp: jsonRes.main.temp,
      humidity: jsonRes.main.humidity,
      weather: jsonRes.weather[0].description,
    };
    
    let tempAlert, tempmsg, humAlert, humMsg, rainAlert, rainMsg;

    if (resObject.temp > 35) {
      tempAlert = "High Temperature:";
      tempmsg = [
        "Avoid excessive watering.",
        "Consider shade nets or mulching to protect crops.",
        " Monitor for heat stress and provide adequate ventilation.",
      ];
    } else if (resObject.temp < 15) {
      tempAlert = "Low Temperature:";
      tempmsg = [
        "Protect crops from frost with row covers or cold frames.",
        "Consider early planting or greenhouse cultivation",
        " Monitor for chilling injury and provide insulation.",
      ];
    } else {
      tempAlert = "Temperature is within a suitable range.";
    }

    if (resObject.humidity > 80) {
      humAlert = "High Humidity:";
      humMsg = [
        "Improve drainage to prevent waterlogging.",
        " Space plants adequately to promote air circulation.",
        " Monitor for fungal diseases and apply appropriate fungicides.",
      ];
    } else if (resObject.humidity < 40) {
      humAlert = "Low Humidity:";
      humMsg = [
        " Increase watering frequency, especially during dry periods.",
        " Consider misting or humidifiers to maintain adequate moisture.",
        " Monitor for wilting and provide appropriate irrigation.",
      ];
    } else {
      humAlert = "Humidity is within a suitable range.";
    }

    if (
      resObject.weather === "Rain" ||
      resObject.weather === "light rain" ||
      resObject.weather === "Heavy rain"
    ) {
      rainAlert = "Rainy Weather";
      rainMsg = [
        "Check for waterlogging and ensure proper drainage.",
        " Monitor for fungal diseases and apply appropriate fungicides.",
      ];
    } else if (resObject.weather == "Snow") {
      rainAlert = "Freezing Weather";
      rainMsg = [
        "Protect crops from heavy snowfall with row covers or cold frames.",
        "Ensure adequate drainage to prevent waterlogging.",
        "Monitor for frost damage and provide insulation.",
      ];
    } 

    // console.log(resObject.weather);

    res.send({
      tempAlert,
      tempmsg,
      humAlert,
      humMsg,
      rainAlert,
      rainMsg,
      city,
    });
  } catch (e) {
    console.log(e);
  }
});

module.exports = router;
