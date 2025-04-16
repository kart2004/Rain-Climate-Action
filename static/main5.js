flatpickr(".datepicker", {});

const choices = new Choices('[data-trigger]',
      {
        searchEnabled: false,
        itemSelectText: '',
      });

function validateForm() {
    var x = document.forms["myForm"]["date"].value;
    if (x == "") {
        //alert("Date must be filled out");
        conditions.innerText = "Please pick a date!";
        return false;
    }
    res = parseInt(x.substring(0, 4), 10);
    if (res<1980) {
        //alert("Date must be after 1980!");
        conditions.innerText = "Please pick a date after year 1980!";
        return false;
    }
    console.log(x);
    return true;
}

document.getElementById("myVideo").playbackRate = 1;
