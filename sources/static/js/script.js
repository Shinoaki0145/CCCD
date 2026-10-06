// Elements for Front side
const frontDropArea = document.querySelector('#front_drag_area') || document.querySelector('.drag-area');
const frontBtn = document.querySelector('#front_btn') || frontDropArea.querySelector('.button');
const frontInput = document.querySelector('#front_input') || frontDropArea.querySelector('input');

// Elements for Back side
const backDropArea = document.querySelector('#back_drag_area');
const backBtn = document.querySelector('#back_btn');
const backInput = document.querySelector('#back_input');

const submitBtn = document.querySelector('#submitbtn');
const resultPerson = document.querySelector('.person');

let file = null;        // Front image
let file_back = null;   // Back image
let dataExtracted = [];

// Sidebar elements
let menu_btn = document.querySelector("#menu-btn");
let sidebar = document.querySelector(".sidebar");
let search_btn = document.querySelector(".bx-search-alt-2");

if (menu_btn) {
  menu_btn.onclick = function() {
    sidebar.classList.toggle("active");
  };
}

if (search_btn) {
  search_btn.onclick = function() {
    sidebar.classList.toggle("active");
  };
}

function loading_on() {
  document.querySelector('.overlay').style.display = "block";
}

function loading_off() {
  document.querySelector('.overlay').style.display = "none";
}

// Front side browse
if (frontBtn && frontInput) {
  frontBtn.onclick = () => { frontInput.click(); };
  frontInput.addEventListener('change', function () {
    file = this.files[0];
    if (file) {
      frontDropArea.classList.add('active');
      displayFile(file, frontDropArea, (val) => { file = val; });
    }
  });
}

// Front side drag & drop
if (frontDropArea) {
  frontDropArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    frontDropArea.classList.add('active');
    const header = frontDropArea.querySelector('.header');
    if (header) header.textContent = 'Release to Upload';
  });

  frontDropArea.addEventListener('dragleave', () => {
    frontDropArea.classList.remove('active');
    const header = frontDropArea.querySelector('.header');
    if (header) header.textContent = 'Drag & Drop';
  });

  frontDropArea.addEventListener('drop', (event) => {
    event.preventDefault();
    file = event.dataTransfer.files[0];
    if (file) {
      displayFile(file, frontDropArea, (val) => { file = val; });
    }
  });
}

// Back side browse
if (backBtn && backInput) {
  backBtn.onclick = () => { backInput.click(); };
  backInput.addEventListener('change', function () {
    file_back = this.files[0];
    if (file_back) {
      backDropArea.classList.add('active');
      displayFile(file_back, backDropArea, (val) => { file_back = val; });
    }
  });
}

// Back side drag & drop
if (backDropArea) {
  backDropArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    backDropArea.classList.add('active');
    const header = backDropArea.querySelector('.header');
    if (header) header.textContent = 'Release to Upload';
  });

  backDropArea.addEventListener('dragleave', () => {
    backDropArea.classList.remove('active');
    const header = backDropArea.querySelector('.header');
    if (header) header.textContent = 'Drag & Drop';
  });

  backDropArea.addEventListener('drop', (event) => {
    event.preventDefault();
    file_back = event.dataTransfer.files[0];
    if (file_back) {
      displayFile(file_back, backDropArea, (val) => { file_back = val; });
    }
  });
}

function displayFile(f, targetArea, setFileCallback) {
  let fileType = f.type;
  let validExtensions = ['image/jpeg', 'image/jpg', 'image/png'];

  if (validExtensions.includes(fileType)) {
    let fileReader = new FileReader();
    fileReader.onload = () => {
      let fileURL = fileReader.result;
      targetArea.innerHTML = `<img src="${fileURL}" alt="Preview" style="max-height: 90%; max-width: 90%; object-fit: contain; border-radius: 12px;" />`;
    };
    fileReader.readAsDataURL(f);
  } else {
    targetArea.classList.remove('active');
    setFileCallback(null);
    Swal.fire({
      icon: 'error',
      title: 'Định dạng không hỗ trợ',
      text: 'Chỉ hỗ trợ file ảnh JPEG, JPG, PNG!'
    });
  }
}

// AJAX request
const xhr = new XMLHttpRequest();
xhr.onreadystatechange = function() {
  if (xhr.readyState == XMLHttpRequest.DONE && xhr.status == 200) {
    loading_off();
    const res = JSON.parse(xhr.responseText);
    const data = res.data || [];
    const issue_date = res.issue_date || '';
    const issue_place = res.issue_place || '';

    const update = new Date();
    const personImg = document.querySelector('.person__img');
    if (personImg && data.length > 0) {
      personImg.innerHTML = `<img src="/static/results/0.jpg?v=${update.getTime()}" />`;
    }

    if (data.length >= 8) {
      document.querySelector('.info__id').innerHTML = `Số (ID): ${data[0]}`;
      document.querySelector('.info__name').innerHTML = `Họ và tên (Full name): ${data[1]}`;
      document.querySelector('.info__date').innerHTML = `Ngày sinh (Date of birth): ${data[2]}`;
      document.querySelector('.info__sex').innerHTML = `Giới tính (Sex): ${data[3]}`;
      document.querySelector('.info__nation').innerHTML = `Quốc tịch (Nationality): ${data[4]}`;
      document.querySelector('.info__hometown').innerHTML = `Quê quán (Place of origin): ${data[5]}`;
      document.querySelector('.info__address').innerHTML = `Nơi thường trú (Place of residence): ${data[6]}`;
      document.querySelector('.info__doe').innerHTML = `Ngày hết hạn (Date of expiry): ${data[7]}`;
    }

    const issueDateEl = document.querySelector('.info__issue_date');
    if (issueDateEl && issue_date) {
      issueDateEl.innerHTML = `Ngày cấp (Date of issue): <b>${issue_date}</b>`;
    }

    const issuePlaceEl = document.querySelector('.info__issue_place');
    if (issuePlaceEl && issue_place) {
      issuePlaceEl.innerHTML = `Nơi cấp (Place of issue): <b>${issue_place}</b>`;
    }

    dataExtracted = [{
      id: data[0] || '',
      name: data[1] || '',
      date_of_birth: data[2] || '',
      sex: data[3] || '',
      nationality: data[4] || '',
      hometown: data[5] ? `"${data[5]}"` : '',
      address: data[6] ? `"${data[6]}"` : '',
      date_of_expiry: data[7] || '',
      date_of_issue: issue_date || '',
      place_of_issue: issue_place ? `"${issue_place}"` : ''
    }];

    if (resultPerson) {
      resultPerson.style.display = "block";
      resultPerson.scrollIntoView({ behavior: 'smooth' });
    }

    Swal.fire({
      icon: 'success',
      title: 'Thành công',
      text: 'Trích xuất thông tin CCCD thành công!',
      footer: `CODE: ${xhr.status}`
    });
  }
  else if (xhr.readyState == XMLHttpRequest.DONE && xhr.status >= 400) {
    loading_off();
    try {
      const data = JSON.parse(xhr.responseText);
      Swal.fire({
        icon: 'error',
        title: 'Oops...',
        text: String(data.message || 'Lỗi xử lý ảnh!'),
        footer: `CODE: ${xhr.status}`
      });
    } catch(err) {
      Swal.fire({
        icon: 'error',
        title: 'Oops...',
        text: 'Có lỗi xảy ra trong quá trình trích xuất!',
        footer: `CODE: ${xhr.status}`
      });
    }
  }
  else if (xhr.readyState == XMLHttpRequest.DONE && xhr.status == 201) {
    downloadCSV({ filename: "cccd_data.csv" });
    Swal.fire({
      toast: true,
      position: 'top-end',
      icon: 'success',
      title: 'Tải file dữ liệu thành công!',
      showConfirmButton: false,
      timer: 3000
    });
  }
};

// Form submit event
if (submitBtn) {
  submitBtn.addEventListener("click", function(e) {
    e.preventDefault();

    if (!file && !file_back) {
      Swal.fire({
        icon: 'warning',
        title: 'Chưa chọn ảnh',
        text: 'Vui lòng tải lên ảnh CCCD mặt trước hoặc mặt sau!'
      });
      return;
    }

    loading_on();
    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    }
    if (file_back) {
      formData.append('file_back', file_back);
    }

    const URL = '/uploader';
    xhr.open('POST', URL, true);
    xhr.send(formData);
  });
}

function convertArrayOfObjectsToCSV(args) {
  const data = args.data;
  if (!data || !data.length) return '';

  const columnDelimiter = args.columnDelimiter || ',';
  const lineDelimiter = args.lineDelimiter || '\n';
  const keys = Object.keys(data[0]);

  let result = '';
  result += keys.join(columnDelimiter);
  result += lineDelimiter;

  data.forEach(item => {
    let ctr = 0;
    keys.forEach(key => {
      if (ctr > 0) result += columnDelimiter;
      result += item[key];
      ctr++;
    });
    result += lineDelimiter;
  });

  return result;
}

function downloadExtracted() {
  const URL = '/download';
  const formData = new FormData();
  formData.append('file', JSON.stringify(dataExtracted));
  xhr.open('POST', URL, true);
  xhr.send(formData);
}

function downloadCSV(args) {
  let csv = convertArrayOfObjectsToCSV({
    data: dataExtracted
  });
  if (!csv) return;

  const filename = args.filename || 'cccd_data.csv';
  if (!csv.match(/^data:text\/csv/i)) {
    csv = 'data:text/csv;charset=utf-8,\uFEFF' + csv; // UTF-8 BOM for Excel Vietnamese display
  }

  const data = encodeURI(csv);
  const link = document.createElement('a');
  link.setAttribute('href', data);
  link.setAttribute('download', filename);
  link.click();
}
