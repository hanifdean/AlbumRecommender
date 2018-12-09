var express = require('express');
var router = express.Router();

/* GET home page. */
router.post('/cluster', function(req, res, next) {
  res.render('cluster', { albumId: req.body['albumId'], clusterId: req.body['clusterId'] });
});

router.get('/', function(req, res, next) {
  res.render('index', { title: 'Albums' });
});

module.exports = router;
