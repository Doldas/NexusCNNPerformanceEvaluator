let formaction_xhr,consolewindow_xhr,testinformationwindow_xhr;
let updateDelay = 30000;

String.prototype.hashCode = function() {
    let hash = 0,
        i, chr;
    if (this.length === 0) return hash;
    for (i = 0; i < this.length; i++) {
        chr = this.charCodeAt(i);
        hash = ((hash << 5) - hash) + chr;
        hash |= 0; // Convert to 32bit integer
    }
    return hash;
}

function byId(id) {
    return document.getElementById(id);
}

function byName(name,index=-1) {
    if(index > -1)
        return document.getElementsByName(name)[index];
    else
        return document.getElementsByName(name);
}

function byClass(classname,index=-1) {
    if(index > -1)
        return document.getElementsByClassName(classname)[index];
    else
        return document.getElementsByClassName(classname);
}

function getFilenameFromUrl() {
    const url = window.location.pathname;
    return url.substring(url.lastIndexOf('/') + 1);
}

function doRegister(e) {
    e.preventDefault();
    if (byId('usrname').value !== "" && byId('psw').value !== "" && byId('psw-repeat').value !== "") {
        const form = byId('register-form');
        formaction_xhr.addEventListener('readystatechange', processRegister, false);
        formaction_xhr.open(form.method, form.action, true);
        // Register Credentials
        const data = new FormData(form);
        // Post The Data To Server.
        formaction_xhr.send(data);
    }
    return false;
}

function processRegister() {
    // If Login Succeeded Fetch The Response And Load The Menu.
    if (formaction_xhr.readyState === XMLHttpRequest.DONE && formaction_xhr.status === 200) {
        formaction_xhr.removeEventListener('readystatechange', processRegister, false);
        window.location.reload();
    }
    // Unauthorized Error to be displayed
    else if (formaction_xhr.readyState === XMLHttpRequest.DONE && formaction_xhr.status === 401 ||
        formaction_xhr.readyState === XMLHttpRequest.DONE && formaction_xhr.status === 400) {
        formaction_xhr.removeEventListener('readystatechange', processRegister, false);
        console.log(JSON.parse(this.responseText).error);
        byId("err").innerHTML = JSON.parse(this.responseText).error;
    } else if (formaction_xhr.status === 404) {
        formaction_xhr.removeEventListener('readystatechange', processRegister, false);
        byId("err").innerHTML = "Login service is unavailable!";
    }
}

function doLogin(e) {
    e.preventDefault();
    if (byId('usrname').value !== "" && byId('psw').value !== "") {
        const form = byId('login-form');
        formaction_xhr.addEventListener('readystatechange', processLogin, false);
        formaction_xhr.open(form.method, form.action, true);
        // Login Credentials
        const data = new FormData(form);
        // Post The Data To Server.
        formaction_xhr.send(data);
    }
    return false;
}

function processLogin() {
    // If Login Succeeded Fetch The Response And Load The Menu.
    if (formaction_xhr.readyState === XMLHttpRequest.DONE && formaction_xhr.status === 200) {
        let responseData = JSON.parse(this.responseText);
        formaction_xhr.removeEventListener('readystatechange', processLogin, false);
        sessionStorage.setItem("user", responseData);
        window.location.reload();
    }
    // Unauthorized Error to be displayed
    else if (formaction_xhr.readyState === XMLHttpRequest.DONE && formaction_xhr.status === 401 ||
        formaction_xhr.readyState === XMLHttpRequest.DONE && formaction_xhr.status === 400) {
        formaction_xhr.removeEventListener('readystatechange', processLogin, false);
        console.log(JSON.parse(this.responseText).error);
        byId("err").innerHTML = JSON.parse(this.responseText).error;
        // Login Page Not Found.
    } else if (formaction_xhr.status === 404) {
        formaction_xhr.removeEventListener('readystatechange', processLogin, false);
        byId("err").innerHTML = "Login service is unavailable!";
    }
}

function doAiToolBoxCommand(e) {
    e.preventDefault();
        const form = byId('aitoolbox-form');
    formaction_xhr.addEventListener('readystatechange', processAiToolBoxCommand, false);
    formaction_xhr.open(form.method, form.action, true);
        // Login Credentials
        const data = new FormData(form);
        data.append("form-btn",e.submitter.value);
        // Post The Data To Server.
    formaction_xhr.send(data);

    return false;
}

function doConsoleWindowUpdate(e) {
    consolewindow_xhr.addEventListener('readystatechange', processConsoleWindowUpdate, false);
    consolewindow_xhr.open("GET","console.php/readconsole", true);
    consolewindow_xhr.send("");

    return false;
}

function doTestInformationWindowUpdate(e) {
    testinformationwindow_xhr.addEventListener('readystatechange', processTestInformationWindowUpdate, false);
    testinformationwindow_xhr.open("GET","aitoolbox.php/getTestCaseInformation", true);
    testinformationwindow_xhr.send("");

    return false;
}

function processConsoleWindowUpdate() {
    if (consolewindow_xhr.readyState === XMLHttpRequest.DONE && consolewindow_xhr.status === 200) {
        let responseData = JSON.parse(this.responseText);
        if(responseData.message.hashCode() !== byId("console").innerHTML.hashCode()){
            byId("console").innerHTML = responseData.message;
        }
        consolewindow_xhr.removeEventListener('readystatechange', processConsoleWindowUpdate, false);
    }
    else if (consolewindow_xhr.readyState === XMLHttpRequest.DONE && consolewindow_xhr.status === 401 ||
        consolewindow_xhr.readyState === XMLHttpRequest.DONE && consolewindow_xhr.status === 400) {
        console.log(JSON.parse(this.responseText).error);
        byId("err").innerHTML = JSON.parse(this.responseText).error;
        consolewindow_xhr.removeEventListener('readystatechange', processConsoleWindowUpdate, false);
    } else if (consolewindow_xhr.status === 404) {
        byId("err").innerHTML = "Console service is unavailable!";
        consolewindow_xhr.removeEventListener('readystatechange', processConsoleWindowUpdate, false);
    }


}

function processTestInformationWindowUpdate() {
    if (testinformationwindow_xhr.readyState === XMLHttpRequest.DONE && testinformationwindow_xhr.status === 200) {
        let responseData = JSON.parse(this.responseText);
        if(responseData.message.hashCode() !== byId("information").innerHTML.hashCode()){
            byId("information").innerHTML = responseData.message;
        }
        testinformationwindow_xhr.removeEventListener('readystatechange', processTestInformationWindowUpdate, false);
    }
    else if (testinformationwindow_xhr.readyState === XMLHttpRequest.DONE && testinformationwindow_xhr.status === 401 ||
        testinformationwindow_xhr.readyState === XMLHttpRequest.DONE && testinformationwindow_xhr.status === 400) {
        console.log(JSON.parse(this.responseText).error);
        byId("err").innerHTML = JSON.parse(this.responseText).error;
        testinformationwindow_xhr.removeEventListener('readystatechange', processTestInformationWindowUpdate, false);
    } else if (testinformationwindow_xhr.status === 404) {
        byId("err").innerHTML = "Console service is unavailable!";
        testinformationwindow_xhr.removeEventListener('readystatechange', processTestInformationWindowUpdate, false);
    }
}

function initConsoleWindow(){
    doConsoleWindowUpdate();
    doTestInformationWindowUpdate();
    setInterval(function() {
        doConsoleWindowUpdate();
        doTestInformationWindowUpdate();
    }, updateDelay);
}

function processAiToolBoxCommand() {
    byId("err").innerHTML = ""

    // If Login Succeeded Fetch The Response And Load The Menu.
    if (formaction_xhr.readyState === XMLHttpRequest.DONE && formaction_xhr.status === 200) {
        let responseData = this.responseText;
        formaction_xhr.removeEventListener('readystatechange', processAiToolBoxCommand, false);

        console.log(responseData);
    }
    // Unauthorized Error to be displayed
    else if (formaction_xhr.readyState === XMLHttpRequest.DONE && formaction_xhr.status !== 404) {
        formaction_xhr.removeEventListener('readystatechange', processAiToolBoxCommand, false);
        console.log(JSON.parse(this.responseText).error);
        byId("err").innerHTML = JSON.parse(this.responseText).error;
    } else if (formaction_xhr.status === 404) {
        formaction_xhr.removeEventListener('readystatechange', processAiToolBoxCommand, false);
        byId("err").innerHTML = "AI ToolBox Service Is Unvailable";
    }
}

function updateSettingsView(e) {
    if(e.target.checked){
        byId("general-nrofrandfilters").disabled=false;
        Array.prototype.forEach.call(byClass("filter-param"),function(el){
            el.disabled=true;
            el.hidden=true;
        });
        Array.prototype.forEach.call(byName("filter-settings"),function(el){
            el.hidden=true;
        });
    }
    else{
        byId("general-nrofrandfilters").disabled=true;
        Array.prototype.forEach.call(byClass("filter-param"),function(el){
            el.disabled=false;
            el.hidden=false;
        });
        Array.prototype.forEach.call(byName("filter-settings"),function(el){
            el.hidden=false;
        });
    }
}

function updateFilterField(e) {
    const rootFieldset = e.target.parentElement.parentElement;

    if(e.target.checked){
        Array.prototype.forEach.call(rootFieldset.children,function(el){
            if(el.type==="fieldset"){
                if(!byId("general-generaterandom").checked){
                    el.hidden=false;
                    Array.prototype.forEach.call(el.children,function(el){
                        if(el.type==="select-one"){
                            el.disabled=false;
                        }

                    });
                }
            }

        });

    }
    else{
        Array.prototype.forEach.call(rootFieldset.children,function(el){
            if(el.type==="fieldset"){
                if(!byId("general-generaterandom").checked){
                    el.hidden=true;
                    Array.prototype.forEach.call(el.children,function(el){
                        if(el.type==="select-one"){
                            el.disabled=true;
                        }

                    });
                }
            }

        });
    }
}

function main() {
    try {
        if (window.XMLHttpRequest) {
            // code for IE7+, Firefox, Chrome, Opera, Safari
            formaction_xhr = new XMLHttpRequest();
            consolewindow_xhr = new XMLHttpRequest();
            testinformationwindow_xhr = new XMLHttpRequest();
        } else {
            throw new Error('Cannot create XMLHttpRequest object');
        }
    } catch (e) {
        alert('"XMLHttpRequest failed!' + e.message);
    }

    if(byId("login-form")){
        byId("login-form").addEventListener('submit', doLogin, false);
    }
    else if(byId("register-form")){
        byId("register-form").addEventListener('submit', doRegister, false);
    }
    
    if(byId("aitoolbox-form")){
        initConsoleWindow();
        byId("general-generaterandom").addEventListener('change', updateSettingsView, false);
        byId("aitoolbox-form").addEventListener('submit', doAiToolBoxCommand, false);
        Array.prototype.forEach.call(byClass("filter"),function(el){
            el.addEventListener("change",updateFilterField);
        });
    }
}

// Connect the main function to window load event
window.addEventListener("load", main, false);