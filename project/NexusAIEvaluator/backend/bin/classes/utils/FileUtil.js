var Promise = require("bluebird");
var fs = Promise.promisifyAll(require("fs"));

module.exports = {
    getFiledataAsString(filepath){
        return fs.readFileSync(filepath,"utf8");
    },
    getFiledataFromJsonToObjects(filepath){
        return JSON.parse(this.getFiledataAsString(filepath));
    }
};