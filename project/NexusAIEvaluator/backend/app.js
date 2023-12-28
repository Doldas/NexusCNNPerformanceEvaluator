var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
var indexRouter = require('./routes/index');
var usersRouter = require('./routes/users');
var session = require('express-session')
var app = express();
var multer = require('multer');
var upload = multer();
var cors = require('cors')
var migration = require('./bin/classes/database/migrations/migration_init');
migration.runMigration();

// view engine setup
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use('/stylesheets/vendor/fontawesome', express.static(__dirname + '/node_modules/@fortawesome/fontawesome-free/'));
app.use('/stylesheets/vendor/bootstrap', express.static(__dirname + '/node_modules/bootstrap/'));
app.use('/javascripts/vendor/bootstrap', express.static(__dirname + '/node_modules/bootstrap/'));
app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
// for parsing multipart/form-data
app.use(upload.array()); 
app.use(express.static('public'));


app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));
app.use(session({
  secret: 'keyboard cat',
  cookie: { maxAge: 900000 },
  resave: true,
  saveUninitialized: true
  }));


app.use('/', indexRouter);
app.use('/users', usersRouter);
// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).render('error', { message: 'Something went wrong!' });
});

const corsOptions = {
  origin: 'http://localhost:4200', // Replace with your Angular app's URL
  methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
  credentials: true, // Enable cookies or other credentials
  optionsSuccessStatus: 204, // Some legacy browsers (IE11, various SmartTVs) choke on 204
};

app.use(cors(corsOptions));

module.exports = app;
