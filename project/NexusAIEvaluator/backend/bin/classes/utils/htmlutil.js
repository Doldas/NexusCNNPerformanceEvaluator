module.exports={
    escapeText(str) {
        return str.replace(
            /[^0-9A-Za-z ]/g,
            ch => "&#" + ch.charCodeAt(0) + ";"
        );
     }     
}