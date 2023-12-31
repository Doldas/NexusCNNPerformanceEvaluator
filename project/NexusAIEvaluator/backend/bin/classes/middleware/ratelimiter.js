const rateLimit = require("express-rate-limit");

const rateLimiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 100, // max 100 requests per windowMs
    message: "You have exceeded your 100 requests per minute limit.",
    headers: true,
  });

  module.exports = rateLimiter;