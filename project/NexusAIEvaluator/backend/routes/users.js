const express = require("express");
const jwt = require('jsonwebtoken')
const router = express.Router();
const htmlUtil = require("../bin/classes/utils/htmlutil");
const accountCtrl = require("../bin/controllers/usercontroller");
const fileUtil = require("../bin/classes/utils/FileUtil");
const ratelimiter = require("../bin/classes/middleware/ratelimiter");
const cors = require('cors');

const corsOptions = {
  origin: 'http://localhost:4200', // Replace with your Angular app's URL
  methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
  credentials: true, // Enable cookies or other credentials
  optionsSuccessStatus: 204, // Some legacy browsers (IE11, various SmartTVs) choke on 204
};

router.use(cors(corsOptions));

router.post("/login",ratelimiter, function (req, res, next) {
  const username = htmlUtil.escapeText(req.body.username.toLowerCase());
  const psw = htmlUtil.escapeText(req.body.password);
  accountCtrl
    .login(username, psw)
    .then((result) => {
      if (result) {
        const user = {
          id:Date.now(),
          username: username,
          password: psw
        };

        jwt.sign({user}, fileUtil.getFiledataFromJsonToObjects("bin/configs/secret.json").jwtsecret,{expiresIn: fileUtil.getFiledataFromJsonToObjects("bin/configs/secret.json").jwtexpiredinseconds}, (err,token)=>{
            const response = {
              status: 200,
              token: token,
              msg: "user is now logged in."
            }
            res.status(200).json(response);
        });   
      } else {
        const response = {
          status: 400,
          error: "Incorrect username or password."
        }
        res.status(400).json(response);
      }
    })
    .catch((err) => {
      const response = {
        status: 400,
        error: "Incorrect username or password."
      }
      res.status(400).json(response);
    });
});

router.get("/isTokenExpired",ratelimiter,verifyToken,function(req,res){
  jwt.verify(req.token, fileUtil.getFiledataFromJsonToObjects("bin/configs/secret.json").jwtsecret, function(err, decoded) {
    if(err){
      res.status(200).json({ isExpired: true });
    }
    else {
      res.status(200).json({ isExpired: false });
    }
  });
});

router.get("/profile",ratelimiter,verifyToken,function (req, res) {

    jwt.verify(req.token, fileUtil.getFiledataFromJsonToObjects("bin/configs/secret.json").jwtsecret, function(err, decoded) {
      if(err){
        res.sendStatus(403);
      }
      else {
        res.status(200).json({ user: decoded });
      }
    });
});

function verifyToken(req,res,next){
  const bearerHeader = req.headers['authorization'];
  if(typeof bearerHeader !== 'undefined'){
    const bearer = bearerHeader.split(' ');
    const bearerToken = bearer[1];
    req.token = bearerToken;
    next();
  } else {
    res.sendStatus(403);
  }
}

module.exports = router;
