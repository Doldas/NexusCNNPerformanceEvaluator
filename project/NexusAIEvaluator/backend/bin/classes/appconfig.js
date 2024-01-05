const FILEUTIL = require("./utils/FileUtil");

module.exports = {
    getConf(){
        try{
          return FILEUTIL.getFiledataFromJsonToObjects("./configs/app.json");
        }
        catch(err){
          console.error("configs/app.json not found.");
        }
        return null;
    }
};