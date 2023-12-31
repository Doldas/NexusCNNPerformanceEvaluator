const express = require('express')
const cors = require('cors');
const router = express.Router();
const ratelimiter = require("../bin/classes/middleware/ratelimiter");

const corsOptions = {
  origin: 'http://localhost:4200', // Replace with your Angular app's URL
  methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
  credentials: true, // Enable cookies or other credentials
  optionsSuccessStatus: 204, // Some legacy browsers (IE11, various SmartTVs) choke on 204
};

router.use(cors(corsOptions));
/* GET home page. */
router.get('/',ratelimiter,async (req, res, next) => {
  const response = {
      status: 200
  }
  res.status(200).json(response);
});

module.exports = router;